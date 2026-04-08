# core/training_data.py

import os
import json
import hashlib
import tempfile
from typing import List, Dict, Any

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import getpass

from .config import CONFIG
from .feedback import load_feedback
from .label_map import normalize_sys_id, resolve_hierarchical_sys_id, normalise_to_taxonomy, load_folder_name_map
from .model_loader import load_current_model
from .databricks_job import trigger_training_job, wait_for_job
from .model_registry import get_model_registry, save_model_registry, get_current_model_version, load_model_by_version
from .training_metadata import next_version_name, get_metadata, log_retrain_event, acquire_lock, release_lock, is_locked, should_auto_promote, get_corrections_since_last_retrain


TRAIN_COLS = ["Title", "Description", "SysID", "RowID"]


# ----------------------------------------------------------------------
# Utility: Deterministic RowID for deduplication
# ----------------------------------------------------------------------
def _row_id(title: str, description: str, sys_id: str) -> str:
    payload = f"{title.strip().lower()}||{description.strip().lower()}||{sys_id.strip().upper()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# ----------------------------------------------------------------------
# 1) Build training data from feedback
# ----------------------------------------------------------------------
def build_training_from_feedback(min_confidence: float = 0.0) -> pd.DataFrame:
    fb = load_feedback()
    if fb.empty:
        return pd.DataFrame(columns=TRAIN_COLS)

    # load_current_model() returns 5-tuple: (None, model, folder_map, valid_sysids, meta)
    _, _, _, valid_sysids, _ = load_current_model()

    rows = []
    for _, r in fb.iterrows():
        title = str(r.get("Title", "") or "")
        desc = str(r.get("Description", "") or "")
        corr = normalize_sys_id(r.get("Correct SysID", ""))
        pred = normalize_sys_id(r.get("Predicted SysID", ""))
        label = corr if corr else pred
        label = resolve_hierarchical_sys_id(label, valid_sysids) or label
        # Normalise to taxonomy (zero-pad, file-level index correction)
        try:
            _fmap = load_folder_name_map()
            if _fmap:
                label = normalise_to_taxonomy(label, set(_fmap.keys()))
        except Exception:
            pass

        if not (title.strip() or desc.strip()):
            continue
        if not label:
            continue

        rows.append({
            "Title": title,
            "Description": desc,
            "SysID": label,
            "RowID": _row_id(title, desc, label),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by="RowID", inplace=True)
        df = df.drop_duplicates(subset=["RowID"]).reset_index(drop=True)

    df.to_csv(CONFIG["training_data_csv"], index=False)
    return df


# ----------------------------------------------------------------------
# 2) Append external training rows
# ----------------------------------------------------------------------
def append_training_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    _, _, _, valid_sysids, _ = load_current_model()

    cleaned = []
    for r in rows:
        title = str(r.get("Title", "") or "")
        desc = str(r.get("Description", "") or "")
        sid = normalize_sys_id(r.get("SysID", ""))
        sid = resolve_hierarchical_sys_id(sid, valid_sysids) or sid
        if not sid:
            continue

        cleaned.append({
            "Title": title,
            "Description": desc,
            "SysID": sid,
            "RowID": _row_id(title, desc, sid),
        })

    path = CONFIG["training_data_csv"]
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=TRAIN_COLS)
    merged = pd.concat([old, pd.DataFrame(cleaned)], ignore_index=True)

    if not merged.empty:
        merged.sort_values(by=["RowID"], inplace=True)
        merged = merged.drop_duplicates(subset=["RowID"]).reset_index(drop=True)

    merged.to_csv(path, index=False)
    return merged


# ----------------------------------------------------------------------
# 3) Queue the training data into MLflow (feedback-only)
# ----------------------------------------------------------------------
def enqueue_training_data_as_mlflow_artifact() -> dict:
    """
    Creates a short MLflow run where the feedback-based training_data.csv
    is logged as an artifact: queued/training_data.csv
    """
    td_local = CONFIG["training_data_csv"]
    if not os.path.exists(td_local):
        raise FileNotFoundError("training_data.csv does not exist. Build training data first.")

    exp = CONFIG.get("mlflow_experiment")
    if not exp:
        raise RuntimeError("Missing MLFLOW_EXPERIMENT configuration.")

    mlflow.set_experiment(exp)

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    with mlflow.start_run(run_name="retrain_queue") as qrun:
        mlflow.set_tag("queue_type", "feedback_training_csv")
        mlflow.set_tag("requested_by", username)
        mlflow.set_tag("source", "streamlit_retrain_page")
        mlflow.log_artifact(td_local, artifact_path="queued")

        queue_run_id = qrun.info.run_id

    return {
        "queue_run_id": queue_run_id,
        "artifact_relpath": "queued/training_data.csv",
    }


# ----------------------------------------------------------------------
# Helpers: manifest download + previous model URI
# ----------------------------------------------------------------------
def _resolve_experiment_id(client: MlflowClient):
    exp_name_or_path = CONFIG.get("mlflow_experiment")
    exp = client.get_experiment_by_name(exp_name_or_path)
    return exp.experiment_id if exp else exp_name_or_path


def _download_manifest_from_latest_full_pipeline():
    client = MlflowClient()
    exp_id = _resolve_experiment_id(client)

    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'FINISHED' and tags.mlflow.runName = 'full_pipeline'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished 'full_pipeline' runs found.")

    parent = runs[0]
    local_dir = tempfile.mkdtemp(prefix="manifest_")
    local_path = client.download_artifacts(
        parent.info.run_id, "model_manifest.json", local_dir
    )

    with open(local_path, "r") as f:
        return json.load(f)


def _resolve_previous_model_uri() -> str | None:
    """
    Get the current model's MLflow URI from the registry, if present.
    This is used to enable incremental training without old raw data.
    """
    try:
        _, _, _, _, meta = load_current_model()
        return meta.get("mlflow_model_uri")
    except Exception:
        return None


# ----------------------------------------------------------------------
# 4) Full retraining via Databricks Job (incremental-aware)
# ----------------------------------------------------------------------
def _register_new_model_version(queue_run_id: str) -> dict:
    """
    After the training job completes, find the best model from the
    latest full_pipeline run and register it in model_registry.json.
    Uses ac_vNNN naming convention with full metadata.
    """
    from datetime import datetime, timezone

    client = MlflowClient()

    exp_name = CONFIG.get("mlflow_experiment")
    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        raise RuntimeError(f"Experiment '{exp_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.mlflow.runName = 'full_pipeline'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished 'full_pipeline' run found after retraining.")

    parent_run = runs[0]
    best_run_id = parent_run.data.tags.get("best_model_run_id")
    if not best_run_id:
        raise RuntimeError("Pipeline run missing 'best_model_run_id' tag.")

    model_uri = f"runs:/{best_run_id}/model"
    pipeline_version = parent_run.data.tags.get("pipeline_version", "unknown")
    best_model_name = parent_run.data.tags.get("best_model", "unknown")

    # ── Sequential version naming (ac_vNNN) ──
    version_name = next_version_name()

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Extract metrics from the best candidate run ──
    try:
        best_run = client.get_run(best_run_id)
        accuracy = best_run.data.metrics.get("accuracy")
        f1_macro = best_run.data.metrics.get("macro_f1")
        n_samples = best_run.data.metrics.get("n_training_samples")
    except Exception:
        accuracy, f1_macro, n_samples = None, None, None

    registry = get_model_registry()
    registry[version_name] = {
        "mlflow_model_uri": model_uri,
        "trained_on": now,
        "model_type": best_model_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "n_training_samples": int(n_samples) if n_samples else None,
        "notes": f"Cumulative retrain from feedback (queue run {queue_run_id})",
        "pipeline_version": pipeline_version,
        "folder_name_map": CONFIG.get("folder_name_map", ""),
        "parent_run_id": parent_run.info.run_id,
        "best_model_run_id": best_run_id,
    }
    save_model_registry(registry)

    return {
        "version": version_name,
        "model_uri": model_uri,
        "training_run_id": best_run_id,
        "parent_run_id": parent_run.info.run_id,
        "pipeline_version": pipeline_version,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "n_training_samples": int(n_samples) if n_samples else None,
    }


def run_retraining_mlflow_via_job(on_step=None, on_poll=None) -> dict:
    """
    Cumulative retraining via Databricks Job with metadata tracking:
    1. Acquire retrain lock (single-run guard)
    2. Enqueue new feedback CSV as MLflow artifact
    3. Pass --previous_model_uri for cumulative merge
    4. Wait for job completion
    5. Register new model version (ac_vNNN naming)
    6. Auto-compare new vs current model, auto-promote if improved
    7. Log retrain event in training_metadata.json
    8. Update dynamic threshold, release lock

    Args:
        on_step: Optional callback(step_name, step_number, total_steps)
        on_poll: Optional callback(life_cycle_state, elapsed_seconds)
    """
    import time as _time

    total_steps = 7
    lock_id = None

    def _step(name, num):
        if on_step:
            on_step(name, num, total_steps)

    try:
        # ── Step 1: Acquire lock ──
        _step("Acquiring retrain lock", 1)
        locked, existing = is_locked()
        if locked:
            raise RuntimeError(
                f"Another retrain is already in progress (lock: {existing}). "
                "Wait for it to finish or check for stale locks."
            )
        success, lock_id = acquire_lock()
        if not success:
            raise RuntimeError("Could not acquire retrain lock.")

        start_time = _time.time()
        new_corrections = get_corrections_since_last_retrain()

        # ── Step 2: Queue training data ──
        _step("Queueing training data as MLflow artifact", 2)
        queue_info = enqueue_training_data_as_mlflow_artifact()
        prev_model_uri = _resolve_previous_model_uri()

        # ── Step 3: Trigger job ──
        _step("Triggering Databricks training job", 3)
        job_id = CONFIG.get("training_job_id")
        if not job_id:
            raise RuntimeError("Missing TRAINING_JOB_ID configuration.")

        python_params = [
            "--queue_run_id", queue_info["queue_run_id"],
            "--queue_artifact_path", queue_info["artifact_relpath"],
        ]
        if prev_model_uri:
            python_params.extend(["--previous_model_uri", prev_model_uri])

        job_run_id = trigger_training_job(python_params)

        # ── Step 4: Wait for completion ──
        _step("Waiting for training job to complete", 4)
        wait_for_job(job_run_id, on_poll=on_poll)

        # ── Step 5: Register new version ──
        _step("Registering new model version", 5)
        reg_info = _register_new_model_version(queue_info["queue_run_id"])

        duration = int(_time.time() - start_time)

        # ── Step 6: Auto-compare and conditionally promote ──
        _step("Comparing new model vs current", 6)
        from .model_registry import get_promoted_version, set_promoted_version

        current_version = get_promoted_version()
        current_accuracy = None
        current_f1 = None

        if current_version:
            registry = get_model_registry()
            current_entry = registry.get(current_version, {})
            current_accuracy = current_entry.get("accuracy")
            current_f1 = current_entry.get("f1_macro")

        new_accuracy = reg_info.get("accuracy")
        new_f1 = reg_info.get("f1_macro")

        promoted = False
        promote_reason = "No accuracy data available for comparison."
        if new_accuracy is not None:
            promoted, promote_reason = should_auto_promote(
                new_accuracy=new_accuracy,
                current_accuracy=current_accuracy,
                new_f1=new_f1,
                current_f1=current_f1,
            )
            if promoted:
                set_promoted_version(reg_info["version"])

        # ── Step 7: Log retrain event ──
        _step("Logging retrain event", 7)
        log_retrain_event(
            version=reg_info["version"],
            dataset_size=reg_info.get("n_training_samples") or 0,
            new_corrections=new_corrections,
            accuracy=new_accuracy,
            f1_macro=new_f1,
            promoted=promoted,
            previous_version=current_version,
            training_run_id=reg_info["training_run_id"],
            parent_run_id=reg_info["parent_run_id"],
            duration_seconds=duration,
            notes=promote_reason,
        )

        return {
            "job_run_id": job_run_id,
            "version": reg_info["version"],
            "model_uri": reg_info["model_uri"],
            "queue_run_id": queue_info["queue_run_id"],
            "training_run_id": reg_info["training_run_id"],
            "pipeline_version": reg_info["pipeline_version"],
            "accuracy": new_accuracy,
            "f1_macro": new_f1,
            "promoted": promoted,
            "promote_reason": promote_reason,
            "duration_seconds": duration,
            "new_corrections": new_corrections,
        }

    finally:
        # Always release lock, even on failure
        if lock_id:
            release_lock(lock_id)


def clear_feedback_and_training() -> dict:
    """
    Delete feedback.csv and training_data.csv to reset the feedback loop.
    Returns summary of what was cleared.
    """
    import os
    cleared = []

    fb_path = _feedback_path_from_config()
    if os.path.exists(fb_path):
        os.remove(fb_path)
        cleared.append("feedback.csv")

    td_path = CONFIG.get("training_data_csv", "")
    if td_path and os.path.exists(td_path):
        os.remove(td_path)
        cleared.append("training_data.csv")

    return {"cleared": cleared}


def _feedback_path_from_config() -> str:
    """Get feedback path from config (mirrors feedback._feedback_path)."""
    return CONFIG.get("feedback_csv") or "feedback.csv"
