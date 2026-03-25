# pages/Diagnostics.py
import streamlit as st
import pandas as pd

from core.file_utils import load_csv_or_excel
from core.model_registry import list_model_versions, load_model_by_version
from core.label_map import normalize_sys_id
from core.feedback import load_feedback
from core.prediction import predict_batch_df

st.set_page_config(page_title="Diagnostics", page_icon="🧪", layout="wide")
st.title("🧪 Diagnostics")

with st.expander("📘 What this page shows", expanded=True):
    st.markdown("""
Use this page to understand:

- **Model-trained labels** (from the model artifact)
- **Taxonomy labels** (from your folder_name_map)
- **Unseen labels in feedback** (labels not learned by the model)

**Behavior**
- If your feedback file already contains **Correct SysID**, we use it automatically.
- If not, you can select which column contains the correct labels.
- Columns containing the word **'predict'** cannot be used as the truth column.
""")


# ============================================================
# Load model version & artifacts
# ============================================================

versions = list_model_versions()
if not versions:
    st.error("No model found. Train or register a model first.")
    st.stop()

selected_version = st.selectbox(
    "Select model version",
    options=versions,
    index=0,
    key="diag_model_version_select",
)

try:
    vectorizer, model, folder_name_map, trained_labels, meta = load_model_by_version(selected_version)
except Exception as e:
    st.error(f"Failed to load version '{selected_version}': {e}")
    st.stop()

taxonomy_labels = set(folder_name_map.keys())
st.caption(f"Loaded version: **{meta.get('version', selected_version)}**")


# ============================================================
# Taxonomy / Training Overview
# ============================================================

st.write("---")
st.subheader("🗂️ Taxonomy & Training Overview")

st.write(f"- Total taxonomy labels: **{len(taxonomy_labels)}**")
st.write(f"- Model trained labels: **{len(trained_labels)}**")

if st.checkbox("Show some taxonomy labels", value=False, key="diag_show_taxonomy_subset"):
    st.write(sorted(list(taxonomy_labels))[:50])

st.write("---")
st.subheader("📚 Labels the model has actually been trained on (ground truth: model artifact)")
coverage_df = pd.DataFrame({
    "SysID": sorted(list(trained_labels)),
    "Folder Name": [folder_name_map.get(s, "Unknown") for s in trained_labels]
})
st.dataframe(coverage_df, use_container_width=True, height=320)

never_seen = sorted(list(taxonomy_labels - trained_labels))
st.markdown("### 📍 Taxonomy labels never seen in training")
if never_seen:
    st.dataframe(
        pd.DataFrame({
            "SysID": never_seen,
            "Folder Name": [folder_name_map.get(s, "Unknown") for s in never_seen]
        }),
        use_container_width=True,
        height=240
    )
else:
    st.success("All taxonomy labels are included in the model training set!")


# ============================================================
# Feedback Unseen Label Detection — Auto-use Correct SysID when present
# ============================================================

st.write("---")
st.subheader("🚩 Unseen Labels in Feedback")

fb = load_feedback()
if fb.empty:
    st.info("No feedback available.")
    st.stop()

# Show feedback preview so users can see what’s on disk
st.write("### 📄 Feedback file preview (first 50 rows)")
st.dataframe(fb.head(50), use_container_width=True)

fb = fb.copy()

# Determine the truth column:
# - Prefer existing "Correct SysID" (produced by Predict page ingestion)
# - Else allow user to select any column except those containing 'predict'
if "Correct SysID" in fb.columns:
    truth_col = "Correct SysID"
    st.success("Using **Correct SysID** column from feedback.")
else:
    st.write("### 🔧 Select the Correct/True SysID Column (not found automatically)")
    all_cols = list(fb.columns)
    forbidden_cols = [c for c in all_cols if "predict" in c.lower()]
    selectable_cols = [c for c in all_cols if c not in forbidden_cols]

    if not selectable_cols:
        st.error("No valid column available for selection (columns containing 'predict' cannot be used).")
        st.stop()

    truth_col = st.selectbox(
        "Choose the column that contains the TRUE / CORRECT SysID:",
        options=selectable_cols,
        key="diag_truth_col_select_fb",
    )
    if not truth_col:
        st.error("You must select a truth label column to continue.")
        st.stop()

# Normalize chosen truth labels
vals = fb[truth_col].astype(str).str.strip().apply(normalize_sys_id)

# Validate similarity to taxonomy / trained labels (warn only)
taxonomy_match_rate = (vals.isin(taxonomy_labels)).mean()
trained_match_rate = (vals.isin(trained_labels)).mean()
THRESHOLD = 0.2
if taxonomy_match_rate < THRESHOLD and trained_match_rate < THRESHOLD:
    st.warning(
        f"The selected truth column '{truth_col}' has low match with taxonomy/trained labels "
        f"(taxonomy: {taxonomy_match_rate:.1%}, trained: {trained_match_rate:.1%}). "
        "If this is not the correct column, please pick another one."
    )

# Additional guard: ensure the selected truth column isn't just the predicted values
if "Predicted SysID" in fb.columns:
    pred_norm = fb["Predicted SysID"].astype(str).str.strip().apply(normalize_sys_id)
    same_rate = (pred_norm == vals).mean()
    if same_rate >= 0.9:
        st.error(
            f"The selected column '{truth_col}' appears to be the same as 'Predicted SysID' "
            f"(normalized equality in {same_rate:.1%} of rows). Please select a different column."
        )
        st.stop()
    elif same_rate >= 0.5:
        st.warning(
            f"The selected column '{truth_col}' matches 'Predicted SysID' in {same_rate:.1%} of rows. "
            "Ensure this is indeed the true label column."
        )

# Unseen by model = not in trained labels
fb["CorrectLabel_norm"] = vals
fb["UnseenByModel"] = ~fb["CorrectLabel_norm"].isin(trained_labels)
unseen_fb = fb[fb["UnseenByModel"]].copy()

st.write(f"Total unseen labels: **{len(unseen_fb)}** (labels not learned by this model)")

if unseen_fb.empty:
    st.success("🎉 No unseen labels — all corrections match trained model labels!")
else:
    cols_to_show = ["Title", "Description", "Predicted SysID", truth_col, "Timestamp"]
    cols_to_show = [c for c in cols_to_show if c in unseen_fb.columns]
    st.dataframe(unseen_fb[cols_to_show].head(200), use_container_width=True, height=320)

    st.download_button(
        "⬇️ Download unseen feedback rows",
        data=unseen_fb.to_csv(index=False).encode("utf-8"),
        file_name="unseen_feedback.csv",
        mime="text/csv",
        key="diag_unseen_download_btn",
    )


# ============================================================
# Optional Evaluation
# ============================================================

st.write("---")
st.subheader("📊 Evaluate on a CSV (optional)")

file = st.file_uploader(
    "Upload CSV for evaluation",
    type=["csv", "xlsx", "xls"],
    key="diag_eval_file_uploader",
)
if file is not None:
    try:
        df = load_csv_or_excel(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    cols = list(df.columns)

    title_col = st.selectbox("Title Column", options=cols, key="diag_eval_title_select")
    desc_col = st.selectbox("Description Column", options=cols, key="diag_eval_desc_select")

    # Show ALL columns for truth selection, but exclude any that contain 'predict'
    eval_forbidden = [c for c in cols if "predict" in c.lower()]
    eval_selectable = [c for c in cols if c not in eval_forbidden]

    if not eval_selectable:
        st.error("No valid truth column available (columns containing 'predict' cannot be used).")
        st.stop()

    truth_col_eval = st.selectbox(
        "True SysID Column (required)",
        options=eval_selectable,
        key="diag_eval_truth_select",
    )

    if st.button("Run Evaluation", key="diag_eval_run_btn"):
        preds = predict_batch_df(
            df=df,
            title_col=title_col,
            desc_col=desc_col,
            vectorizer=vectorizer,
            model=model,
            folder_name_map=folder_name_map,
            valid_sysids=trained_labels,
            true_sysid_col=truth_col_eval,
        )

        if preds.empty:
            st.warning("No valid rows to evaluate.")
        else:
            st.dataframe(preds.head(25), use_container_width=True)

            if "Match" in preds.columns:
                acc = float((preds["Match"] == True).mean()) * 100.0
                st.success(f"Accuracy: **{acc:.2f}%**")
            else:
                st.info("Truth column missing; unable to compute accuracy.")