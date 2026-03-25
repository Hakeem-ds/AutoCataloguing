# core/model_loader.py

import streamlit as st
import mlflow
import os

os.environ["DATABRICKS_HOST"] = os.getenv('DATABRICKS_HOST')
os.environ["DATABRICKS_TOKEN"] = os.getenv('DATABRICKS_TOKEN')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')

mlflow.set_tracking_uri("databricks")

from .model_registry import (
    get_current_model_version,
    load_model_by_version
)

@st.cache_resource
def load_current_model():
    """
    Loads the latest available model using Streamlit caching.
    Always returns:
        model_pipeline,
        folder_name_map,
        valid_sysids,
        meta
    """

    # ✔ Ensure MLflow loads via Databricks REST API (NO CLI fallback)
    mlflow.set_tracking_uri("databricks")

    version = get_current_model_version()
    if version is None:
        return None, {}, set(), {}

    # Unified 4-return-value interface
    _, model, folder_name_map, valid_sysids, meta = load_model_by_version(version)

    return model, folder_name_map, valid_sysids, meta