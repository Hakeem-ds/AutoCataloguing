# core/training_data.py

import os
import hashlib
from typing import List, Dict, Any
import tempfile
import json
import pandas as pd
from datetime import datetime
import getpass

import mlflow
from mlflow.tracking import MlflowClient

from .config import CONFIG
from .feedback import load_feedback
from .label_map import normalize_sys_id, resolve_hierarchical_sys_id
from .model_loader import load_current_model
from .databricks_job import trigger_training_job, wait_for_job
from .model_registry import get_model_registry, save_model_registry


TRAIN_COLS = ["Title", "Description", "SysID", "RowID"]


def enqueue_training_data_as_mlflow_artifact() -> dict:
    """
    Logs artifacts/training_data.csv as an MLflow artifact into a short 'queue' run
    and returns {'queue_run_id', 'artifact_relpath'}.
    """
    td_local = CONFIG["training_data_csv"]
    if not os.path.exists(td_local):
        raise FileNotFoundError("training_data.csv does not exist. Build training data first.")

    # Ensure tracking/experiment are set
    mlflow.set_tracking_uri(CONFIG.get("mlflow_tracking_uri", "databricks"))
    if not CONFIG.get("mlflow_experiment"):
        raise RuntimeError("Missing MLFLOW_EXPERIMENT (env or secrets).")
    mlflow.set_experiment(CONFIG["mlflow_experiment"])

    artifact_relpath = "queued/training_data.csv"
    username = None
    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    with mlflow.start_run(run_name="retrain_queue") as qrun:
        mlflow.set_tag("queue_type", "feedback_training_csv")
        mlflow.set_tag("requested_by", username)
        mlflow.set_tag("source", "streamlit_retrain_page")
        mlflow.set_tag("pipeline_version", "queued")  # optional
        mlflow.log_artifact(td_local, artifact_path="queued")

        queue_run_id = qrun.info.run_id

    return {"queue_run_id": queue_run_id, "artifact_relpath": artifact_relpath}



def _row_id(title: str, description: str, sys_id: str) -> str:
    """Deterministic row ID for deduplication."""
    payload = f"{title.strip().lower()}||{description.strip().lower()}||{sys_id.strip().upper()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# ============================================================
# 1) Build training data from feedback (UNCHANGED)
# ============================================================

def build_training_from_feedback(min_confidence: float = 0.0) -> pd.DataFrame:
    fb = load_feedback()
    if fb.empty:
        return pd.DataFrame(columns=TRAIN_COLS)

    _, _, _, valid_sysids, _ = load_current_model()
    rows = []

    for _, r in fb.iterrows():
        title = str(r.get("Title", "") or "")
        desc = str(r.get("Description", "") or "")
        corr = normalize_sys_id(r.get("Correct SysID", ""))
        pred = normalize_sys_id(r.get("Predicted SysID", ""))

        label = corr if corr else pred
        label = resolve_hierarchical_sys_id(label, valid_sysids) or label
        if not (title.strip() or desc.strip()):
            continue
        if not label:
            continue

        rows.append({
            "Title": title,
            "Description": desc,
            "SysID": label,
            "RowID": _row_id(title, desc, label)
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=["RowID"], inplace=True)
        df = df.drop_duplicates(subset=["RowID"]).reset_index(drop=True)

    df.to_csv(CONFIG["training_data_csv"], index=False)
    return df


# ============================================================
# 2) Append external training rows (UNCHANGED)
# ============================================================

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
            "RowID": _row_id(title, desc, sid)
        })

    path = CONFIG["training_data_csv"]
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=TRAIN_COLS)
    merged = pd.concat([old, pd.DataFrame(cleaned)], ignore_index=True)

    if not merged.empty:
        merged.sort_values(by=["RowID"], inplace=True)
        merged = merged.drop_duplicates(subset=["RowID"]).reset_index(drop=True)

    merged.to_csv(path, index=False)
    return merged


# ============================================================
# 3) MLflow retraining via Databricks Job (CORRECT VERSION)
# ============================================================

def _resolve_experiment_id(client: MlflowClient):
    exp_name_or_path = CONFIG.get("mlflow_experiment")
    if not exp_name_or_path:
        raise RuntimeError("Missing MLFLOW_EXPERIMENT (env or secrets).")

    exp = client.get_experiment_by_name(exp_name_or_path)
    return exp.experiment_id if exp else exp_name_or_path


def _download_manifest_from_latest_full_pipeline():
    """
    After the job completes, fetch latest full_pipeline parent run,
    download its model_manifest.json from MLflow artifacts.
    """
    mlflow.set_tracking_uri(CONFIG.get("mlflow_tracking_uri", "databricks"))
    client = MlflowClient()
    exp_id = _resolve_experiment_id(client)

    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'FINISHED' and tags.mlflow.runName = 'full_pipeline'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No finished 'full_pipeline' runs found in MLflow experiment.")

    parent = runs[0]
    local_dir = tempfile.mkdtemp(prefix="manifest_")

    try:
        local_path = client.download_artifacts(parent.info.run_id, "model_manifest.json", local_dir)
        with open(local_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to download model_manifest.json: {e}")


def run_retraining_mlflow_via_job():
    """
    - Enqueues training_data.csv as MLflow artifact into a 'queue' run
    - Triggers Databricks job (main.py --from_step train ... --use_feedback --queue_run_id --queue_artifact_path)
    - Waits for completion
    - Fetches model_manifest.json from MLflow artifacts (latest 'full_pipeline' run)
    - Registers new MLflow model version locally
    """
    # 0) Enqueue
    q = enqueue_training_data_as_mlflow_artifact()
    queue_run_id = q["queue_run_id"]
    artifact_relpath = q["artifact_relpath"]  # e.g. 'queued/training_data.csv'

    # 1) Trigger job with queue params
    job_run_id = trigger_training_job(
        python_params=[
            "--from_step", "train",
            "--use_feedback",
            "--queue_run_id", queue_run_id,
            "--queue_artifact_path", artifact_relpath,
        ]
    )

    # 2) Wait for job
    wait_for_job(job_run_id)

    # 3) Fetch latest manifest from MLflow
    manifest = _download_manifest_from_latest_full_pipeline()
    best = manifest["best"]
    new_model_uri    = best["model_uri"]
    training_run_id  = best["run_id"]
    version          = f"svm_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # 4) Update registry
    reg = get_model_registry()
    reg[version] = {
        "mlflow_model_uri": new_model_uri,
        "trained_on": datetime.utcnow().isoformat(),
        "notes": f"Retrained via Databricks Job from queued run={queue_run_id}; training run={training_run_id}",
        "folder_name_map": CONFIG["folder_name_map"],
        "pipeline_version": manifest.get("pipeline_version"),
    }
    save_model_registry(reg)

    return {
        "version": version,
        "job_run_id": job_run_id,
        "queue_run_id": queue_run_id,
        "training_run_id": training_run_id,
        "model_uri": new_model_uri,
        "pipeline_version": manifest.get("pipeline_version"),
    }