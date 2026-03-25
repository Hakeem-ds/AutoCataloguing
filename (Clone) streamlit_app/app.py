import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Document Folder Prediction App",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Sidebar Logo
# --------------------------
logo_path = Path("assets/logo.png")
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
st.sidebar.caption("App version: 1.0.0 • Powered by SVM + TF‑IDF")

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
- Correct predictions and feed them back into the learning loop  
- Diagnose label coverage, unseen SysIDs, and taxonomy structure  
- Manage the taxonomy directly (with parent enforcement + audit log)  
- Retrain models safely using deduplicated, user‑verified feedback  
- Version, compare and validate models over time  

The interface is built to support **non‑technical users**:
- Clear language  
- Guided steps  
- Automatic guardrails and validation  
- Hierarchical SysID handling  
- Transparent audit logs  
"""
)

st.write("---")

# --------------------------
# How To Use Section
# --------------------------
with st.expander("📘 How to Use This App"):
    st.markdown(
        """
### 1️⃣ Predict
Go to **Predict** and enter a *Title* + *Description* or upload a CSV.  
You’ll receive:
- The predicted **SysID**  
- The mapped **Folder Name**  
- The model’s **confidence**  

You can also download CSVs for manual correction.

---

### 2️⃣ Send Corrections (Feedback)
Correct predictions directly on the Predict page, or upload a corrected CSV.  
The system automatically:
- Normalizes SysIDs  
- Resolves hierarchical paths  
- Removes duplicates  

All corrections are stored safely.

---

### 3️⃣ Diagnose the Model
Under **Diagnostics**, you can:
- See which SysIDs are used in training  
- Detect never‑seen taxonomy entries  
- Find unseen SysIDs in feedback  
- Evaluate model accuracy via CSV upload  

---

### 4️⃣ Manage the Taxonomy
Under **Taxonomy Manager**, you can:
- Add new SysIDs  
- Auto‑create missing parents  
- Import CSVs of new folders  
- Edit folder names  
- Add aliases  
- View an audit trail of every change  
- Export the taxonomy (JSON or CSV)

---

### 5️⃣ Retrain the Model
Under **Review & Retrain**, follow the guided steps:
1. Load or upload feedback  
2. Build clean training data  
3. Retrain an SVM model  
4. Automatically version the model  
5. Review accuracy and metrics  

A new version is saved under `models/versioned/`.

---

### 6️⃣ Compare Versions
Go to **Model Versions** to:
- Inspect model metadata  
- Load/smoke test versions  
- Add notes  
- Confirm which version is currently active  
"""
    )

# --------------------------
# Footer
# --------------------------
st.write("---")
st.markdown(
    """
<p style='text-align:center;color:grey;font-size:0.95rem;'>
Built for the Corporate Archives — Powered by Streamlit & classic ML  
</p>
""",
    unsafe_allow_html=True,
)