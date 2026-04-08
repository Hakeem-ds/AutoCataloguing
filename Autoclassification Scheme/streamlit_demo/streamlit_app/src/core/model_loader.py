# core/model_loader.py

import logging
from typing import Any, Dict, Set, Tuple, Optional
import streamlit as st

from .model_registry import (
    get_current_model_version,
    load_model_by_version,
    sync_from_mlflow,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def load_current_model() -> Tuple[Optional[Any], Any, Dict[str, str], Set[str], Dict[str, Any]]:
    """
    Load the 'current' model version as determined by the registry ordering.
    Auto-syncs from MLflow on first load if no versions are registered yet.

    Returns (ALWAYS 5 values for backward compatibility):
        - vectorizer (None for MLflow models; kept for legacy BC)
        - model_pipeline (sklearn Pipeline)
        - folder_name_map (dict: sys_id -> folder name)
        - valid_sysids (set of labels learned by the model)
        - meta (dict with metadata: version, mlflow_model_uri, notes, etc.)

    Cached across Streamlit runs via st.cache_resource.
    MLflow tracking URI is set once at startup in main.py.
    """
    version = get_current_model_version()

    # Auto-sync: if no versions registered, discover from MLflow
    if version is None:
        logger.info("No model versions in registry — auto-syncing from MLflow...")
        try:
            result = sync_from_mlflow()
            added = result.get("added", [])
            errors = result.get("errors", [])
            if added:
                logger.info("Auto-sync added %d version(s): %s", len(added), added)
            if errors:
                logger.warning("Auto-sync errors: %s", errors)
            version = get_current_model_version()
        except Exception as e:
            logger.warning("Auto-sync failed: %s", e)

    if version is None:
        return None, None, {}, set(), {}

    return load_model_by_version(version)


@st.cache_resource
def load_model_for_version(version: str) -> Tuple[Optional[Any], Any, Dict[str, str], Set[str], Dict[str, Any]]:
    """
    Load a specific model version by name (e.g., 'unversioned', 'svm_v20260225_150000', etc.)

    Returns (ALWAYS 5 values for backward compatibility):
        - vectorizer (None for MLflow models; kept for legacy BC)
        - model_pipeline (sklearn Pipeline)
        - folder_name_map (dict: sys_id -> folder name)
        - valid_sysids (set of labels learned by the model)
        - meta (dict with metadata: version, mlflow_model_uri, notes, etc.)
    """
    return load_model_by_version(version)


def clear_model_caches():
    """
    Clear Streamlit resource caches for model loading.
    Useful after retraining or registry updates.
    """
    try:
        load_current_model.clear()           # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        load_model_for_version.clear()       # type: ignore[attr-defined]
    except Exception:
        pass
