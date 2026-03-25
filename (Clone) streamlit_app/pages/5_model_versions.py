import os
import streamlit as st
import pandas as pd

from core.model_registry import (
    get_model_registry,
    save_model_registry,
    list_model_versions,
    load_model_by_version
)
from core.model_loader import load_current_model


# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="Model Versions", page_icon="📦", layout="wide")
st.title("📦 Model Versions")


with st.expander("📘 About this page", expanded=True):
    st.markdown("""
This page shows all **model versions** saved in the model registry.

### What you can do:
- View metadata & artifact paths  
- Edit & save notes for each version  
- Smoke-test loading a version  
- See which version is currently active  
- Inspect model artifacts for integrity  
    
Models are stored locally under:  models/versioned/
and tracked in: artifacts/model_registry.json""")
    
# ============================================================
# Load registry + detect current model
# ============================================================
versions = list_model_versions()
if not versions:
    st.info("No model versions found yet. Train a model first.")
    st.stop()

# Current model (chosen by registry based on latest trained_on)
_, _, _, _, current_meta = load_current_model()
current_version = current_meta.get("version", "unversioned")


st.subheader(f"Current model in use: **{current_version}**")

registry = get_model_registry()


# ============================================================
# Show each version
# ============================================================
for v in versions:

    # Registry entry (may be incomplete if user manually edited files)
    meta = registry.get(v, {}) if isinstance(registry, dict) else {}
    trained_on = meta.get("trained_on", "N/A")
    notes = meta.get("notes", "")
    vec_path = meta.get("vectorizer_path", "N/A")
    mdl_path = meta.get("model_path", "N/A")
    fmap_path = meta.get("folder_name_map", "N/A")
    fmap_csv = meta.get("folder_mapping_csv", "N/A")

    with st.expander(f"🔹 Version: {v}", expanded=(v == current_version)):

        st.markdown(f"""
### 🗂 Metadata
- **Trained on:** `{trained_on}`
- **Vectorizer file:** `{vec_path}`
- **Model file:** `{mdl_path}`
- **Folder Name Map:** `{fmap_path}`
- **Folder Mapping CSV:** `{fmap_csv}`
""")

        # -------------------------
        # Artifact integrity check
        # -------------------------
        missing = []
        for p in [vec_path, mdl_path]:
            if not (p != "N/A" and os.path.exists(p)):
                missing.append(p)

        if missing:
            st.error(f"⚠ Missing files for this version: {missing}")
        else:
            st.success("All required artifacts are present.")

        st.write("---")

        # -------------------------
        # NOTES EDITOR
        # -------------------------
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
            st.success("Notes saved successfully.")

        st.write("---")

        # -------------------------
        # SMOKE TEST LOADER
        # -------------------------
        st.markdown("#### 🔍 Smoke Test")
        if st.button(f"Load {v}", key=f"load_{v}"):
            try:
                vec, mdl, fmap, valid, meta_v = load_model_by_version(v)
                st.success(f"""
Successfully loaded **{v}**

- Taxonomy entries: **{len(fmap)}**  
- Valid SysIDs: **{len(valid)}**  
- Model classes: **{len(getattr(mdl, 'classes_', [])) if hasattr(mdl, 'classes_') else 'N/A'}**
""")
            except Exception as e:
                st.error(f"Error loading {v}: {e}")