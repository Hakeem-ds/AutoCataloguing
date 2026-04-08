import os
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Document Folder Prediction App",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Load custom CSS if present
# --------------------------
css_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css"))
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# --------------------------
# Sidebar Logo
# --------------------------
logo_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png"))
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_column_width=True)

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("📌 Navigation")
st.sidebar.write(
    """
Use the pages below to:

- 🔮 **Predict** folder SysIDs for documents  
- 🧪 **Diagnose** model behaviour & taxonomy coverage  
- 🗂️ **Manage Taxonomy** (SysIDs, folder names, bulk import)  
- 🔁 **Review & Retrain** models using feedback  
- 📦 **Compare Model Versions**  
"""
)

st.sidebar.write("---")
st.sidebar.caption("App version: 3.1 • Powered by SVM + TF‑IDF • MLflow")

# --------------------------
# Main Title
# --------------------------
st.title("📂 Document Folder Prediction Dashboard")

st.markdown(
    """
Welcome to the **Document Folder Prediction App** — a streamlined tool designed to help the
Corporate Archives automatically classify and route documents to the correct **SysID folder**
using machine learning.

This system supports a full end‑to‑end workflow:

### 🧭 What You Can Do Here
- Predict folder SysIDs for new or existing documents  
- Review **confidence bands** (LOW / MODERATE / HIGH) and **related context** suggestions  
- Correct predictions and feed them back into the learning loop  
- Diagnose label coverage, unseen SysIDs, and taxonomy structure  
- Manage the taxonomy directly (with parent enforcement + audit log)  
- Retrain models via Databricks Jobs (serverless, ~1 minute)  
- Version, compare, promote, and smoke‑test models  

The interface is built to support **non‑technical users**:
- Clear language and guided steps  
- Automatic guardrails and validation  
- Hierarchical SysID handling  
- Transparent audit logs  
"""
)

st.write("---")

# --------------------------
# Getting Started (model not loaded)
# --------------------------
with st.expander("🚀 First Time? Getting Started", expanded=False):
    st.markdown(
        """
### If the model is not loaded yet

On a fresh deployment the model registry may be empty. Follow these steps:

1. Go to **🧬 Model Versions** (sidebar)  
2. Click **🔄 Sync from MLflow** — this discovers all trained models from the MLflow experiment  
3. Select the model you want to use from the table  
4. Click **⭐ Promote this version** to make it the active model  
5. Return to **🔮 Predict** — the model is now loaded and ready  

> **Tip:** The app tries to auto‑sync on first load. If that fails (e.g. network
> timeout), use the manual sync button on the Model Versions page.

### Secrets / Configuration

The app connects to Databricks MLflow for model loading and retraining.  
These values must be set as **Streamlit secrets** (Settings → Secrets) or in a `.env` file:

| Secret | Example |
|---|---|
| `DATABRICKS_HOST` | `https://adb-xxxx.xx.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | `dapi...` (Personal Access Token) |
| `TRAINING_JOB_ID` | `922802212201609` |
| `MLFLOW_EXPERIMENT` | `/Users/.../experiments/ac_model_v2` |
| `MLFLOW_TRACKING_URI` | `databricks` |
"""
    )

# --------------------------
# How To Use Section
# --------------------------
with st.expander("📘 How to Use This App"):
    st.markdown(
        """
### 1️⃣ Predict
Go to **Predict** and enter a *Title* + *Description*, or upload a **CSV / Excel** file.  
You’ll receive:
- The predicted **SysID** and mapped **Folder Name**  
- A **confidence score** with colour‑coded band:  
  - 🟢 **HIGH** (≥92%) — model is very confident  
  - 🟡 **MODERATE** (23–92%) — review suggested  
  - 🔴 **LOW** (<23%) — likely wrong, check suggestions  
- **Related Context** (for MODERATE / LOW predictions):  
  - Top related folders from the probability distribution  
  - **Neighbourhood suggestions** — parent‑level groups with combined confidence  
  - Sibling classes under the same folder hierarchy  
  - A review hint with confidence‑based guidance  

For batch predictions, upload a CSV or Excel file. Results include a downloadable
file with predictions, confidence, bands, and related context.

---

### 2️⃣ Send Corrections (Feedback)
Correct predictions directly on the Predict page using the **correction dropdown**
(pre‑populated with related context suggestions), or upload a corrected CSV.  
The system automatically:
- Normalises SysIDs (uppercase, hyphens, zero‑pad matching)  
- Resolves hierarchical paths against the taxonomy  
- Deduplicates entries  

All corrections are stored in `feedback.csv` and used for the next retrain.

---

### 3️⃣ Diagnose the Model
Under **Diagnostics**, you can:
- See which SysIDs are used in training vs the full taxonomy  
- Detect never‑seen taxonomy entries (coverage gaps)  
- Find unseen SysIDs in feedback  
- Evaluate model accuracy by uploading a labelled CSV  

---

### 4️⃣ Manage the Taxonomy
Under **Taxonomy Manager**, you can:
- **Browse & search** all 1,700+ SysID entries  
- **Add / update** entries with automatic parent enforcement  
- **Bulk import** a CSV with preview and validation  
- **Quick‑rename** folder names  
- **Manage aliases** (alternative labels for the same SysID)  
- View an **audit trail** of every change  
- **Export** the taxonomy as JSON or CSV  

---

### 5️⃣ Review & Retrain
Under **Review & Retrain**, follow the guided steps:
1. Load existing feedback or upload new corrections  
2. Build clean, deduplicated training data  
3. Trigger a **Databricks serverless job** to retrain (~1 minute)  
4. The new model is automatically logged to MLflow with full metrics  
5. Go to **Model Versions** to promote the new model  

---

### 6️⃣ Model Versions
Go to **Model Versions** to:
- **Sync from MLflow** — discover newly trained models  
- **Compare metrics** across versions (accuracy, F1, precision, recall)  
- **Promote** a version to become the active production model  
- **Smoke test** any version with a sample prediction  
- **Inspect internals** (TF‑IDF vocabulary, calibration, model size)  
- **Generate a Model Card** with all metadata  

> **Model not loading?** Click 🔄 Sync from MLflow, then ⭐ Promote a version.
"""
    )

# --------------------------
# Footer
# --------------------------
st.write("---")
st.markdown(
    """
<p style='text-align:center;color:grey;font-size:0.95rem;'>
Built for the Corporate Archives — Powered by Streamlit, scikit‑learn & MLflow<br/>
<em>Shareable link: no Databricks login required</em>
</p>
""",
    unsafe_allow_html=True,
)
