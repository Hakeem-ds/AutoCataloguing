import os

# config.py is inside streamlit_app/core/
# -> app root is one level above this file
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Directories
    "artifacts_dir": os.path.join(APP_ROOT, "artifacts"),
    "models_dir": os.path.join(APP_ROOT, "models"),
    "versioned_models_dir": os.path.join(APP_ROOT, "models", "versioned"),

    # Base (unversioned) model artifacts
    "vectorizer": os.path.join(APP_ROOT, "models", "tfidf_vectorizer.pkl"),
    "svm_model": os.path.join(APP_ROOT, "models", "svm_model.pkl"),

    # Taxonomy & metadata
    "folder_name_map": os.path.join(APP_ROOT, "artifacts", "folder_name_map.json"),
    "folder_mapping_csv": os.path.join(APP_ROOT, "artifacts", "folder_mapping.csv"),

    # Feedback and training data
    "feedback_csv": os.path.join(APP_ROOT, "artifacts", "feedback.csv"),
    "training_data_csv": os.path.join(APP_ROOT, "artifacts", "training_data.csv"),
    "invalid_labels_csv": os.path.join(APP_ROOT, "artifacts", "invalid_labels.csv"),
    "pending_sysids_csv": os.path.join(APP_ROOT, "artifacts", "pending_sysids.csv"),

    # Model registry JSON
    "model_registry_json": os.path.join(APP_ROOT, "artifacts", "model_registry.json"),
}