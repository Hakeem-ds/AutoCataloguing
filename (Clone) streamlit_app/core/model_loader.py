import streamlit as st
from .model_registry import get_current_model_version, load_model_by_version

@st.cache_resource
def load_current_model():
    """
    Loads the latest available model version using Streamlit caching.
    Returns:
        vectorizer, model, folder_name_map, valid_sysids (TRAINED), meta
    """
    version = get_current_model_version()
    if version is None:
        return None, None, {}, set(), {}

    return load_model_by_version(version)