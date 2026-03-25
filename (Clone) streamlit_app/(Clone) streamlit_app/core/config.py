# core/config.py
import os
import os.path as P

# Optional: load from .env if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# If running inside Streamlit, secrets may exist
try:
    import streamlit as st
    SECRETS = getattr(st, "secrets", {})
except Exception:
    SECRETS = {}

APP_ROOT = P.dirname(P.dirname(P.abspath(__file__)))

def _get(k, default=None):
    # priority: env -> secrets -> default
    return os.getenv(k) or (SECRETS.get(k) if isinstance(SECRETS, dict) else None) or default

CONFIG = {
    "app_root": APP_ROOT,

    # Local artifacts registry for the app (used by Streamlit)
    "artifacts_dir": P.join(APP_ROOT, "artifacts"),
    "models_dir": P.join(APP_ROOT, "models"),
    "versioned_models_dir": P.join(APP_ROOT, "models", "versioned"),

    # Legacy pickle locations (still kept for BC; not used by MLflow flow)
    "vectorizer": P.join(APP_ROOT, "models", "tfidf_vectorizer.pkl"),
    "svm_model": P.join(APP_ROOT, "models", "svm_model.pkl"),

    # Taxonomy
    "folder_name_map": P.join(APP_ROOT, "artifacts", "folder_name_map.json"),
    "folder_mapping_csv": P.join(APP_ROOT, "artifacts", "folder_mapping.csv"),

    # Feedback / training data
    "feedback_csv": P.join(APP_ROOT, "artifacts", "feedback.csv"),
    "training_data_csv": P.join(APP_ROOT, "artifacts", "training_data.csv"),
    "invalid_labels_csv": P.join(APP_ROOT, "artifacts", "invalid_labels.csv"),
    "pending_sysids_csv": P.join(APP_ROOT, "artifacts", "pending_sysids.csv"),

    # Model registry JSON (app-local registry)
    "model_registry_json": P.join(APP_ROOT, "artifacts", "model_registry.json"),
}

# ---- Databricks / MLflow integration (normalized to lowercase keys)
CONFIG["databricks_host"]   = _get("DATABRICKS_HOST")
CONFIG["databricks_token"]  = _get("DATABRICKS_TOKEN")
# Accept both env/secrets; cast to int when used
CONFIG["training_job_id"]   = _get("TRAINING_JOB_ID")
# MLflow experiment (path or name), e.g. "/Users/.../experiments/ac_model"
CONFIG["mlflow_experiment"] = _get("MLFLOW_EXPERIMENT")

# Optional: if needed elsewhere, export MLflow tracking URI
CONFIG["mlflow_tracking_uri"] = _get("MLFLOW_TRACKING_URI", "databricks")