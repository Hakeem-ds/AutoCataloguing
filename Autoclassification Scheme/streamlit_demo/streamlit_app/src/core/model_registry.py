# core/model_registry.py

import os
import json
import logging
from typing import Any, Dict, Tuple, Optional, Set, List

import pandas as pd
import mlflow

from .config import CONFIG
from .label_map import normalize_sys_id
from .file_utils import atomic_write

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _file_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_abs(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(_APP_ROOT, path))


def _safe_load_folder_name_map(path: Optional[str]) -> Dict[str, str]:
    if not _file_exists(path):
        return {}
    try:
        data = _load_json(path)
        return {normalize_sys_id(k): v for k, v in data.items()}
    except Exception:
        return {}


def _safe_build_valid_sysids(folder_name_map: Dict[str, str],
                             folder_mapping_csv: Optional[str]) -> Set[str]:
    if folder_name_map:
        return set(folder_name_map.keys())

    if _file_exists(folder_mapping_csv):
        try:
            df = pd.read_csv(folder_mapping_csv)
            col = None
            for c in ["sys_id", "sysid", "id", "folder_id"]:
                if c in df.columns:
                    col = c
                    break
            if col:
                return set(normalize_sys_id(x) for x in df[col].dropna().astype(str).tolist())
        except Exception:
            pass

    return set()


def _extract_trained_labels(model: Any) -> Set[str]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return set(normalize_sys_id(c) for c in classes)

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        for _, step in reversed(list(named_steps.items())):
            step_classes = getattr(step, "classes_", None)
            if step_classes is not None:
                return set(normalize_sys_id(c) for c in step_classes)

    return set()


# ---------------------------------------------------------------------
# Registry I/O + Promotion support
# ---------------------------------------------------------------------

def get_model_registry() -> dict:
    path = CONFIG["model_registry_json"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        atomic_write(path, {"__meta__": {}})
        return {"__meta__": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"__meta__": {}}
        if "__meta__" not in data or not isinstance(data["__meta__"], dict):
            data["__meta__"] = {}
        return data
    except Exception:
        return {"__meta__": {}}


def save_model_registry(registry: dict):
    atomic_write(CONFIG["model_registry_json"], registry)


def get_promoted_version() -> Optional[str]:
    reg = get_model_registry()
    return reg.get("__meta__", {}).get("promoted")


def set_promoted_version(version: Optional[str]) -> None:
    reg = get_model_registry()
    reg.setdefault("__meta__", {})
    if version is None:
        reg["__meta__"].pop("promoted", None)
    else:
        if version not in reg:
            raise KeyError(f"Version '{version}' not found in registry.")
        # Track previous promoted for retention policy (rollback slot)
        old_promoted = reg["__meta__"].get("promoted")
        if old_promoted and old_promoted != version:
            reg["__meta__"]["previous_promoted"] = old_promoted
        reg["__meta__"]["promoted"] = version
    save_model_registry(reg)


def list_model_versions() -> List[str]:
    reg = get_model_registry()
    promoted = reg.get("__meta__", {}).get("promoted")

    items = [(k, v) for k, v in reg.items() if k != "__meta__" and isinstance(v, dict)]
    items.sort(key=lambda kv: kv[1].get("trained_on") or "", reverse=True)
    versions = [k for k, _ in items]

    if promoted and promoted in versions:
        versions.remove(promoted)
        versions.insert(0, promoted)

    return versions


def get_current_model_version() -> Optional[str]:
    reg = get_model_registry()
    promoted = reg.get("__meta__", {}).get("promoted")
    if promoted:
        return promoted

    versions = list_model_versions()
    if versions:
        return versions[0]

    return None


# ---------------------------------------------------------------------
# Retention Policy
# ---------------------------------------------------------------------

def enforce_retention(max_versions: Optional[int] = None) -> dict:
    """
    Enforce the model retention policy.

    Keeps (in priority order):
      1. The promoted version (production)
      2. The previous promoted version (rollback safety net)
      3. The most recently trained versions, up to *max_versions* total

    Versions outside this keep-set are **removed from model_registry.json**.
    MLflow run artifacts are NOT deleted (manual cleanup if storage is a
    concern; the run IDs are returned in the 'archived' list).

    Returns {"kept": [...], "archived": [...]}.
    """
    if max_versions is None:
        max_versions = CONFIG.get("retention_max_versions", 3)

    registry = get_model_registry()
    meta = registry.get("__meta__", {})
    promoted = meta.get("promoted")
    previous = meta.get("previous_promoted")

    # Collect all version entries (not __meta__)
    versions = []
    for k, v in registry.items():
        if k == "__meta__" or not isinstance(v, dict):
            continue
        versions.append((k, v.get("trained_on", "")))

    # Sort by trained_on descending (most recent first)
    versions.sort(key=lambda x: x[1], reverse=True)

    # Build keep set: promoted and previous always kept
    keep = set()
    if promoted and promoted in {v[0] for v in versions}:
        keep.add(promoted)
    if previous and previous in {v[0] for v in versions}:
        keep.add(previous)

    # Fill remaining slots with most recent versions
    for ver_name, _ in versions:
        if len(keep) >= max_versions:
            break
        keep.add(ver_name)

    # Archive everything not in keep set
    archived = []
    for ver_name, _ in versions:
        if ver_name not in keep:
            archived.append(ver_name)
            del registry[ver_name]

    # Clean up previous_promoted if it was archived
    if previous and previous in archived:
        registry.setdefault("__meta__", {}).pop("previous_promoted", None)

    if archived:
        save_model_registry(registry)
        logger.info(
            "Retention: kept %d version(s), archived %d: %s",
            len(keep), len(archived), archived,
        )

    return {"kept": sorted(keep), "archived": archived}


# ---------------------------------------------------------------------
# Unified version loader (MLflow-first)
# ---------------------------------------------------------------------


def _load_model_from_artifacts(model_uri: str) -> Any:
    """Load sklearn model by downloading MLflow artifacts directly.
    Works with mlflow-skinny (no mlflow.sklearn module needed).
    Falls back for Streamlit Cloud where full mlflow is not installed."""
    import cloudpickle as _cp

    client = mlflow.tracking.MlflowClient()

    if model_uri.startswith("runs:/"):
        parts = model_uri.replace("runs:/", "").split("/", 1)
        run_id = parts[0]
        artifact_path = parts[1] if len(parts) > 1 else "model"
    else:
        raise ValueError(f"Unsupported model URI format: {model_uri}")

    local_dir = client.download_artifacts(run_id, artifact_path)
    model_pkl = os.path.join(local_dir, "model.pkl")

    with open(model_pkl, "rb") as f:
        return _cp.load(f)

def _load_mlflow_entry(
    version: str, entry: dict
) -> Tuple[Optional[Any], Any, Dict[str, str], Set[str], Dict[str, Any]]:
    """Shared loader for any registry entry that has an mlflow_model_uri."""
    model_uri = entry["mlflow_model_uri"]
    # Try full mlflow first (Databricks Apps), fallback to manual (Streamlit Cloud)
    try:
        model_pipeline = mlflow.sklearn.load_model(model_uri)
    except (ImportError, AttributeError):
        model_pipeline = _load_model_from_artifacts(model_uri)

    fmap_path = _to_abs(entry.get("folder_name_map") or CONFIG.get("folder_name_map"))
    folder_name_map = _safe_load_folder_name_map(fmap_path)

    valid_sysids = _extract_trained_labels(model_pipeline)
    if not valid_sysids:
        valid_sysids = _safe_build_valid_sysids(folder_name_map, CONFIG.get("folder_mapping_csv"))

    meta = {
        "version": version,
        "mlflow_model_uri": model_uri,
        "folder_name_map": fmap_path,
        "classes": list(valid_sysids),
        "trained_on": entry.get("trained_on"),
        "notes": entry.get("notes"),
        "pipeline_version": entry.get("pipeline_version"),
    }
    return None, model_pipeline, folder_name_map, valid_sysids, meta


def load_model_by_version(version: str) -> Tuple[Optional[Any], Any, Dict[str, str], Set[str], Dict[str, Any]]:
    """
    ALWAYS returns a 5-tuple (backward compatible):
        vectorizer=None,
        model_pipeline,
        folder_name_map,
        valid_sysids,
        meta
    """
    logger.debug("load_model_by_version: %s", version)

    registry = get_model_registry()
    entry = registry.get(version)

    # MLflow entry (versioned or unversioned)
    if entry and "mlflow_model_uri" in entry:
        return _load_mlflow_entry(version, entry)

    # If we reached here, version not found
    raise KeyError(f"Model version '{version}' not found in registry.")


# ---------------------------------------------------------------------
# MLflow Discovery: auto-register all trained models
# ---------------------------------------------------------------------

def sync_from_mlflow() -> dict:
    """
    Scan the MLflow experiment for all finished candidate runs that have
    a logged model, and register any that are missing from model_registry.json.
    Also purges registry entries whose MLflow runs no longer exist.
    Applies retention policy after syncing.

    Returns {"added": [...], "skipped": [...], "purged": [...], "errors": [...],
            "retention": {"kept": [...], "archived": [...]}}.
    """
    from datetime import datetime, timezone

    client = mlflow.tracking.MlflowClient()

    exp_name = CONFIG.get("mlflow_experiment")
    if not exp_name:
        return {"added": [], "skipped": [], "purged": [], "errors": [
            "MLFLOW_EXPERIMENT not configured. Set it in .env or Streamlit secrets."
        ], "retention": {}}

    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        return {"added": [], "skipped": [], "purged": [], "errors": [
            f"Experiment '{exp_name}' not found"
        ], "retention": {}}

    registry = get_model_registry()

    # --- Purge entries whose MLflow runs no longer exist ---
    purged = []
    for k, v in list(registry.items()):
        if k == "__meta__" or not isinstance(v, dict):
            continue
        uri = v.get("mlflow_model_uri", "")
        if uri.startswith("runs:/"):
            run_id = uri.split("/")[1]
            try:
                client.get_run(run_id)
            except Exception:
                purged.append(k)
                del registry[k]
                logger.info("Purged stale registry entry '%s' (run %s deleted).", k, run_id)

    # If promoted version was purged, clear it
    promoted = registry.get("__meta__", {}).get("promoted")
    if promoted and promoted in purged:
        registry["__meta__"].pop("promoted", None)
        logger.warning("Promoted version '%s' was purged (MLflow run deleted).", promoted)

    # If previous_promoted was purged, clear it
    previous = registry.get("__meta__", {}).get("previous_promoted")
    if previous and previous in purged:
        registry["__meta__"].pop("previous_promoted", None)

    # Build set of already-registered run IDs for fast lookup
    registered_run_ids = set()
    for k, v in registry.items():
        if k == "__meta__" or not isinstance(v, dict):
            continue
        uri = v.get("mlflow_model_uri", "")
        if uri.startswith("runs:/"):
            registered_run_ids.add(uri.split("/")[1])

    # Find all candidate runs with models
    candidates = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'candidate::%' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )

    added, skipped, errors = [], [], []

    for run in candidates:
        run_id = run.info.run_id

        if run_id in registered_run_ids:
            skipped.append(run_id)
            continue

        try:
            artifacts = list(client.list_artifacts(run_id, "model"))
            if not artifacts:
                continue
        except Exception:
            continue

        parent_id = run.data.tags.get("mlflow.parentRunId", "")
        run_name = run.data.tags.get("mlflow.runName", "candidate")
        model_name = run_name.replace("candidate::", "")
        training_mode = run.data.tags.get("training_mode", "")

        pipeline_version = "unknown"
        if parent_id:
            try:
                parent_run = client.get_run(parent_id)
                pipeline_version = parent_run.data.tags.get("pipeline_version", "unknown")
                if not training_mode:
                    training_mode = parent_run.data.tags.get("training_mode", "fresh")
            except Exception:
                pass

        start_ts = run.info.start_time / 1000
        trained_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        trained_str = trained_dt.strftime("%Y%m%d_%H%M%S")

        # Use sequential naming (ac_vNNN) if training_metadata module is available
        try:
            from .training_metadata import next_version_name
            version_name = next_version_name()
        except Exception:
            # Fallback to timestamp naming if metadata module not available
            version_name = f"{model_name}_v{trained_str}"
            if version_name in registry:
                version_name = f"{version_name}_{run_id[:8]}"

        try:
            model_uri = f"runs:/{run_id}/model"

            # Pull metrics from the candidate run
            metrics = run.data.metrics or {}
            accuracy = metrics.get("accuracy")
            f1_macro = metrics.get("macro_f1")
            n_samples = metrics.get("n_training_samples")
            n_classes = metrics.get("n_classes")

            # Pull normalisation info from parent run
            label_norm = None
            classes_before = None
            classes_after = None
            if parent_id:
                try:
                    parent_run = client.get_run(parent_id)
                    label_norm = parent_run.data.tags.get("label_normalisation")
                    classes_before = parent_run.data.metrics.get("classes_before_norm")
                    classes_after = parent_run.data.metrics.get("classes_after_norm")
                    if n_samples is None:
                        n_samples = parent_run.data.metrics.get("n_training_samples")
                except Exception:
                    pass

            entry = {
                "mlflow_model_uri": model_uri,
                "trained_on": trained_str,
                "notes": f"Auto-discovered from MLflow ({training_mode})",
                "pipeline_version": pipeline_version,
                "folder_name_map": CONFIG.get("folder_name_map", ""),
                "parent_run_id": parent_id,
                "best_model_run_id": run_id,
            }

            # Add metrics (only if available — don't store None)
            if accuracy is not None:
                entry["accuracy"] = round(accuracy, 4)
            if f1_macro is not None:
                entry["f1_macro"] = round(f1_macro, 4)
            if n_samples is not None:
                entry["n_samples"] = int(n_samples)
            if classes_after is not None:
                entry["n_classes"] = int(classes_after)
            elif n_classes is not None:
                entry["n_classes"] = int(n_classes)
            if label_norm:
                entry["label_normalisation"] = label_norm
            if classes_before is not None:
                entry["classes_before_norm"] = int(classes_before)

            registry[version_name] = entry
            added.append(version_name)
        except Exception as e:
            errors.append(f"{run_id}: {e}")

    if added or purged:
        save_model_registry(registry)

    # --- Apply retention policy ---
    retention = enforce_retention()

    return {
        "added": added,
        "skipped": skipped,
        "purged": purged,
        "errors": errors,
        "retention": retention,
    }
