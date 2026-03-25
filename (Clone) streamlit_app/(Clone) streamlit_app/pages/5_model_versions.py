import os
import streamlit as st
import pandas as pd

from core.model_registry import (
    get_model_registry,
    save_model_registry,
    list_model_versions,
    load_model_by_version,
)
from core.model_loader import load_current_model

st.set_page_config(page_title="Model Versions", page_icon="📦", layout="wide")
st.title("📦 Model Versions")

with st.expander("📘 About this page", expanded=True):
    st.markdown("""
This page shows all **MLflow model versions** stored in the model registry.

### What you can do:
- View MLflow model URIs  
- Edit notes  
- Smoke-test loading a version  
- See the currently active model  
""")

# -------------------------------------------------------------
# Load versions
# -------------------------------------------------------------
versions = list_model_versions()
if not versions:
    st.info("No model versions found yet.")
    st.stop()

_, _, _, current_meta = load_current_model()
current_version = current_meta.get("version", "unversioned")
st.subheader(f"Current active model: **{current_version}**")

registry = get_model_registry()

# -------------------------------------------------------------
# Loop through versions
# -------------------------------------------------------------
for v in versions:
    meta = registry.get(v, {})
    trained_on = meta.get("trained_on", "N/A")
    notes = meta.get("notes", "")

    mlflow_uri = meta.get("mlflow_model_uri", None)

    with st.expander(f"🔹 Version: {v}", expanded=(v == current_version)):
        st.markdown(f"""
### 🗂 Metadata
- **Trained on:** `{trained_on}`
- **MLflow Model URI:** `{mlflow_uri or "Not available"}`
""")

        st.write("---")

        # -----------------------------------------------------
        # Notes editor
        # -----------------------------------------------------
        st.markdown("#### ✏️ Notes")
        new_notes = st.text_area(
            f"Notes for {v}",
            value=notes or "",
            height=120,
            key=f"notes_{v}"
        )
        if st.button(f"Save Notes ({v})", key=f"save_notes_{v}"):
            meta["notes"] = new_notes
            registry[v] = meta
            save_model_registry(registry)
            st.success("Notes saved.")

        st.write("---")

        # -----------------------------------------------------
        # Smoke test
        # -----------------------------------------------------
        st.markdown("#### 🔍 Smoke Test Load")
        if st.button(f"Load {v}", key=f"load_{v}"):
            try:
                _, mdl, fmap, valid, meta_v = load_model_by_version(v)
                st.success(f"""
                Loaded version **{v}**

                - Taxonomy: **{len(fmap)} entries**  
                - Valid SysIDs: **{len(valid)}**  
                - Classes: **{len(getattr(mdl, 'classes_', []))}**
                """)
            except Exception as e:
                st.error(f"Error loading model {v}: {e}")