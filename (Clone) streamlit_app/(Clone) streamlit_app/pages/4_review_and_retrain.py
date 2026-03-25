import os
import io
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from core.file_utils import load_csv_or_excel
from core.config import CONFIG
from core.feedback import append_feedback_rows, load_feedback
from core.training_data import build_training_from_feedback, append_training_rows, run_retraining_mlflow_via_job
from core.model_registry import get_model_registry, save_model_registry
from core.model_loader import load_current_model

st.set_page_config(page_title="Review & Retrain", page_icon="🔁", layout="wide")
st.title("🔁 Review & Retrain")

with st.expander("📘 Flow overview", expanded=True):
    st.markdown("""
**This page guides you through four simple steps:**

1) **Review/Add Feedback** — Upload or add corrections (duplicates automatically removed)  
2) **Build Training Data** — Convert feedback into a clean training file  
3) **Retrain Model** — Train an SVM + TF‑IDF on the training set  
4) **Register Version** — Save the model with a new version and update the registry

Everything uses **SysIDs** (no label encoders) with hierarchical resolution.
""")

# ----------------------------------------------------------
# Step 1 — Feedback
# ----------------------------------------------------------
st.header("1) 📝 Review / Add Feedback")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Feedback (deduplicated)")
    fb = load_feedback()
    if fb.empty:
        st.info("No feedback yet.")
    else:
        st.write(f"Total feedback rows (deduplicated): **{len(fb)}**")
        st.dataframe(fb.tail(25), use_container_width=True, height=280)

with col2:
    st.subheader("Add Feedback (CSV)")
    st.markdown("CSV columns expected: **Title**, **Description**, **Predicted SysID**, **Correct SysID** (optional)")
    up = st.file_uploader("Upload Feedback CSV", type=["csv", "xlsx", "xls"], key="fb_uploader")
    if up is not None:
        try:
            df = load_csv_or_excel(up)
            rows = df.to_dict(orient="records")
            before = len(load_feedback())
            deduped = append_feedback_rows(rows)
            after = len(deduped)
            st.success(f"Feedback saved. Total rows now: **{after}** (removed **{before + len(rows) - after}** duplicates).")
        except Exception as e:
            st.error(f"Failed to append feedback: {e}")

st.write("---")

# ----------------------------------------------------------
# Step 2 — Training set
# ----------------------------------------------------------
st.header("2) 🧰 Build Training Data from Feedback")

c1, c2 = st.columns([2, 1])
with c1:
    if st.button("Build training_data.csv now"):
        before_len = 0
        if os.path.exists(CONFIG["training_data_csv"]):
            try:
                before_len = len(pd.read_csv(CONFIG["training_data_csv"]))
            except Exception:
                before_len = 0

        train_df = build_training_from_feedback()
        if train_df.empty:
            st.warning("No training data produced. Ensure you have feedback rows with labels.")
        else:
            removed = max(0, before_len + len(train_df) - len(train_df.drop_duplicates(subset=['RowID'])))
            st.success(
                f"Training data built with **{len(train_df)}** rows "
                f"(duplicates automatically removed)."
            )
            st.dataframe(train_df.head(25), use_container_width=True, height=280)
            st.session_state["last_train_preview"] = train_df.head(200)

    if "last_train_preview" in st.session_state:
        st.download_button(
            "Download latest training preview (CSV)",
            data=st.session_state["last_train_preview"].to_csv(index=False).encode("utf-8"),
            file_name="training_preview.csv",
            mime="text/csv",
        )

with c2:
    st.subheader("Add external training rows (optional)")
    st.markdown("CSV columns required: **Title**, **Description**, **SysID**")
    ext = st.file_uploader("Upload extra training CSV", type=["csv", "xlsx", "xls"], key="train_uploader")
    if ext is not None:
        try:
            tdf = load_csv_or_excel(ext)
            rows = tdf.to_dict(orient="records")
            merged = append_training_rows(rows)
            st.success(f"Appended. Training set now has **{len(merged)}** rows (deduplicated).")
        except Exception as e:
            st.error(f"Failed to append training rows: {e}")

st.write("---")

# ----------------------------------------------------------
# Step 3 — Retrain Model (MLflow Pipeline)
# ----------------------------------------------------------
st.header("3) 🤖 Retrain Model")

if st.button("Start retraining", key="retrain_btn"):
    try:
        # NEW MLflow retraining
        results = run_retraining_mlflow_via_job()

        st.success(f"MLflow retraining complete.")

        st.success(
            f"New model version: **{results['version']}**\n\n"
            f"MLflow Run ID: `{results['run_id']}`\n\n"
            f"Model URI: `{results['model_uri']}`"
        )
        st.balloons()

        st.write("### 📄 Details")
        st.json(results)

    except Exception as e:
        st.error(f"Retraining failed:\n{e}")