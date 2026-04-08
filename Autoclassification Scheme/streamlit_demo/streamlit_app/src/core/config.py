import os
import os.path as P

try:
    import streamlit as st
    SECRETS = getattr(st, "secrets", {})
except Exception:
    SECRETS = {}


def _get(key: str, default=None):
    v_env = os.getenv(key)
    if v_env not in (None, ""):
        return v_env
    if isinstance(SECRETS, dict):
        v_sec = SECRETS.get(key)
        if v_sec not in (None, ""):
            return v_sec
    return default


APP_ROOT = P.dirname(P.dirname(P.abspath(__file__)))

CONFIG = {
    "app_root": APP_ROOT,
    "artifacts_dir": P.join(APP_ROOT, "artifacts"),
    "models_dir": P.join(APP_ROOT, "models"),
    "versioned_models_dir": P.join(APP_ROOT, "models", "versioned"),
    "folder_name_map": P.join(APP_ROOT, "artifacts", "folder_name_map.json"),
    "folder_mapping_csv": P.join(APP_ROOT, "artifacts", "folder_mapping.csv"),
    "label_map_json": P.join(APP_ROOT, "artifacts", "label_map.json"),
    "feedback_csv": P.join(APP_ROOT, "artifacts", "feedback.csv"),
    "training_data_csv": P.join(APP_ROOT, "artifacts", "training_data.csv"),
    "model_registry_json": P.join(APP_ROOT, "artifacts", "model_registry.json"),
    "training_metadata_json": P.join(APP_ROOT, "artifacts", "training_metadata.json"),
}

CONFIG["databricks_host"] = _get("DATABRICKS_HOST")
CONFIG["databricks_token"] = _get("DATABRICKS_TOKEN")
CONFIG["training_job_id"] = _get("TRAINING_JOB_ID")
CONFIG["mlflow_experiment"] = _get(
    "MLFLOW_EXPERIMENT",
    "/Users/hakeemfujah@tfl.gov.uk/experiments/ac_model_v2",
)
CONFIG["mlflow_tracking_uri"] = _get("MLFLOW_TRACKING_URI", "databricks")

# Confidence bands — derived from calibration curve (see Model Comparison notebook).
# LOW  = model is guessing (< 23% acc in this band)
# HIGH = model is confident and almost always right (99.7% acc)
# These defaults can be overridden by the user in the predict page UI.
CONFIG["confidence_bands"] = {
    "low_upper": 0.23,    # below this = LOW
    "high_lower": 0.92,   # at or above this = HIGH
}

# Model retention: max active versions to keep in the registry.
# Slots: promoted (production) + previous (rollback) + candidate (latest trained).
CONFIG["retention_max_versions"] = 3

# Training metadata & auto-retrain settings.
CONFIG["retrain_settings"] = {
    "base_threshold": 200,      # initial retrain threshold (new corrections needed)
    "min_threshold": 50,        # floor — never require fewer than this
    "max_threshold": 1000,      # ceiling — never require more than this
    "growth_factor": 0.1,       # scales with dataset size & retrain count
    "stale_lock_hours": 2,      # lock auto-expires after this many hours
}
