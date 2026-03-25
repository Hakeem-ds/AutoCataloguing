# pages/Predict.py
from datetime import datetime

import streamlit as st
import pandas as pd

from core.file_utils import load_csv_or_excel
from core.model_registry import list_model_versions, load_model_by_version
from core.prediction import predict_single, predict_batch_df
from core.feedback import append_feedback_rows
from core.label_map import normalize_sys_id, resolve_hierarchical_sys_id


st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("🔮 Document Folder Prediction")

with st.expander("📘 How this works", expanded=True):
    st.markdown("""
This page supports **three independent workflows**:

### 1) Single Prediction
Type a Title + Description and get one prediction.

### 2) Batch Prediction
Upload a file, select **Title**, **Description**, and **Correct Label** columns, run predictions, and (optionally) ingest corrections from the same file.

### 3) Corrections Only (no prediction required)
Upload *any* corrected CSV/XLSX, select **Title**, **Description**, and the **Correct Label** column, and ingest to feedback.

**Notes**
- Any column name can be used for the **Correct Label**; columns containing **'predict'** cannot be selected as truth.
- Only relevant fields are stored in feedback: Title, Description, Predicted SysID, Correct SysID, Model Version, Timestamp.
""")

# ============================================================
# Model Version Selection (NEW)
# ============================================================
versions = list_model_versions()
if not versions:
    st.error("❌ No model versions found. Please train or register a model first.")
    st.stop()

selected_version = st.selectbox(
    "🔧 Model version to use",
    options=versions,
    index=0,
    key="model_version_selector",
)

model, folder_name_map, trained_labels, meta = load_model_by_version(selected_version)

# ============================================================
# 1) Single Prediction
# ============================================================
st.write("---")
st.header("🧍 Single Prediction")

c1, c2 = st.columns([1, 2])

with c1:
    s_title = st.text_input("📄 Title", key="sp_title")
    s_desc = st.text_area("📝 Description", height=120, key="sp_desc")

    if st.button("🔮 Predict", key="sp_btn"):
        html, sysid, folder = predict_single(
            input_title=s_title,
            input_description=s_desc,
            model=model,
            folder_name_map=folder_name_map,
            valid_sysids=trained_labels,
            theme="light",
        )
        st.session_state["sp_html"] = html
        st.session_state["sp_sysid"] = sysid
        st.session_state["sp_folder"] = folder
        st.session_state["sp_title_v"] = s_title
        st.session_state["sp_desc_v"] = s_desc

with c2:
    if "sp_html" in st.session_state:
        st.markdown(st.session_state["sp_html"], unsafe_allow_html=True)

if "sp_html" in st.session_state:
    st.subheader("✏️ Suggest a Correction")
    corr_val = st.text_input("Correct SysID", key="sp_corr")
    if st.button("Submit Correction", key="sp_corr_btn"):
        corr_norm = normalize_sys_id(corr_val)
        corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm
        fb = [{
            "Title": st.session_state.get("sp_title_v", ""),
            "Description": st.session_state.get("sp_desc_v", ""),
            "Predicted SysID": st.session_state.get("sp_sysid", ""),
            "Correct SysID": corr_norm,
            "Model Version": meta.get("version", "unversioned"),
            "Timestamp": datetime.utcnow().isoformat(),
        }]
        summary = append_feedback_rows(fb)
        st.success(
            f"✓ Correction submitted. "
            f"(Skipped blank: {summary['skipped_empty_correct']}, "
            f"Skipped predicted==correct: {summary['skipped_equal_to_pred']})"
        )

# ============================================================
# 2) Batch Prediction (Independent)
# ============================================================
st.write("---")
st.header("📦 Batch Prediction")

pred_file = st.file_uploader(
    "Upload CSV/XLSX for prediction (and/or ingest corrections from the same file)",
    type=["csv", "xlsx", "xls"],
    key="bp_file",
)
if pred_file:
    df_pred = load_csv_or_excel(pred_file)
    st.subheader("📄 File Preview")
    st.dataframe(df_pred.head(20), use_container_width=True)

    cols_pred = list(df_pred.columns)

    st.subheader("🪄 Step A — Select Columns")
    bp_title_col = st.selectbox("Title Column", cols_pred, key="bp_title_col")
    bp_desc_col = st.selectbox("Description Column", cols_pred, key="bp_desc_col")

    # Correct label selection is part of Step A
    forbidden_cols = [c for c in cols_pred if "predict" in c.lower()]
    selectable_corr_cols = [c for c in cols_pred if c not in forbidden_cols]
    if not selectable_corr_cols:
        st.warning("No valid **Correct Label** column available (columns containing 'predict' cannot be used).")
        bp_corr_col = None
    else:
        bp_corr_col = st.selectbox(
            "Correct Label Column (for feedback ingest; any name allowed)",
            selectable_corr_cols,
            key="bp_corr_col",
        )

    st.subheader("🪄 Step B — Run Prediction (optional)")
    if st.button("🔮 Predict Entire File", key="bp_predict_btn"):
        preds = predict_batch_df(
            df=df_pred,
            title_col=bp_title_col,
            desc_col=bp_desc_col,
            model=model,
            folder_name_map=folder_name_map,
            valid_sysids=trained_labels,
            true_sysid_col=None,
        )

        # Attach the selected correct label column to the prediction results for preview & download
        if bp_corr_col:
            preds["Correct SysID"] = df_pred[bp_corr_col].astype(str)
        else:
            preds["Correct SysID"] = ""

        st.session_state["bp_preds"] = preds
        st.success(f"Predicted {len(preds)} rows.")

    if isinstance(st.session_state.get("bp_preds"), pd.DataFrame):
        preds = st.session_state["bp_preds"]
        st.subheader("📊 Predictions")
        st.dataframe(preds.head(25), use_container_width=True)

        # Download predictions for correction (include Correct SysID if user selected a column)
        dl = preds.copy()
        if "Correct SysID" not in dl.columns:
            dl["Correct SysID"] = ""
        st.download_button(
            "⬇️ Download predictions for correction",
            dl.to_csv(index=False).encode("utf-8"),
            "predictions_for_correction.csv",
            "text/csv",
            key="bp_download_template",
        )

    st.subheader("🪄 Step C — Ingest Corrections From This SAME File (optional)")
    if bp_corr_col is None:
        st.info("Select a **Correct Label Column** above to enable ingest from this file.")
    else:
        # Quick validation of the selected correct label column
        vals_norm = df_pred[bp_corr_col].astype(str).str.strip().apply(normalize_sys_id)
        taxonomy_labels = set(folder_name_map.keys())
        match_tax = (vals_norm.isin(taxonomy_labels)).mean()
        match_trn = (vals_norm.isin(trained_labels)).mean()
        if match_tax < 0.2 and match_trn < 0.2:
            st.warning(
                f"⚠ Column '{bp_corr_col}' has low match with taxonomy/trained labels "
                f"(taxonomy: {match_tax:.1%}, trained: {match_trn:.1%})."
            )

        if st.button("📤 Ingest Corrections From This File", key="bp_ingest_btn"):
            fb_rows = []
            preds_df = st.session_state.get("bp_preds")  # may be None

            if isinstance(preds_df, pd.DataFrame) and not preds_df.empty and "Source Index" in preds_df.columns:
                # Use predictions for Predicted SysID; map Correct SysID via Source Index to original df_pred
                for _, prow in preds_df.iterrows():
                    src_idx = prow.get("Source Index", None)
                    if src_idx is None or src_idx not in df_pred.index:
                        continue

                    raw_corr = df_pred.at[src_idx, bp_corr_col] if bp_corr_col in df_pred.columns else ""
                    if pd.isna(raw_corr) or str(raw_corr).strip() == "":
                        continue

                    corr_norm = normalize_sys_id(str(raw_corr).strip())
                    corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm

                    pred_val = prow.get("Predicted SysID", "")
                    pred_norm = normalize_sys_id(str(pred_val)) if pred_val else ""
                    if corr_norm == pred_norm:
                        continue

                    fb_rows.append({
                        "Title": prow.get("Title", ""),
                        "Description": prow.get("Description", ""),
                        "Predicted SysID": pred_val,
                        "Correct SysID": corr_norm,  # ← user's selection applied
                        "Model Version": meta.get("version", "unversioned"),
                        "Timestamp": datetime.utcnow().isoformat(),
                    })
            else:
                # No predictions made; ingest using the current file alone (Predicted may be blank or present)
                for _, row in df_pred.iterrows():
                    raw_corr = row.get(bp_corr_col, "")
                    if pd.isna(raw_corr) or str(raw_corr).strip() == "":
                        continue

                    corr_norm = normalize_sys_id(str(raw_corr).strip())
                    corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm

                    pred_val = row.get("Predicted SysID", "")
                    pred_norm = normalize_sys_id(str(pred_val)) if pred_val else ""
                    if corr_norm == pred_norm:
                        continue

                    fb_rows.append({
                        "Title": row.get(bp_title_col, ""),
                        "Description": row.get(bp_desc_col, ""),
                        "Predicted SysID": pred_val,
                        "Correct SysID": corr_norm,  # ← user's selection applied
                        "Model Version": meta.get("version", "unversioned"),
                        "Timestamp": datetime.utcnow().isoformat(),
                    })

            if fb_rows:
                summary = append_feedback_rows(fb_rows)
                st.success(
                    f"Uploaded {summary['appended']} corrections. "
                    f"(Skipped blank: {summary['skipped_empty_correct']}, "
                    f"Skipped predicted==correct: {summary['skipped_equal_to_pred']})"
                )
            else:
                st.warning("⚠ No valid correction rows found in this file.")

# ============================================================
# 3) Corrections Only (Independent — NOT nested)
# ============================================================
st.write("---")
st.header("✅ Corrections Only (No Prediction Required)")

corr_file = st.file_uploader(
    "Upload corrected CSV/XLSX (must contain TRUE/Correct SysIDs)",
    type=["csv", "xlsx", "xls"],
    key="co_file",
)
if corr_file:
    df_corr = load_csv_or_excel(corr_file)
    st.subheader("📄 Corrected File Preview")
    st.dataframe(df_corr.head(20), use_container_width=True)

    cols_corr = list(df_corr.columns)

    st.subheader("🪄 Step 1 — Select Columns")
    co_title_col = st.selectbox("Title Column (corrected file)", cols_corr, key="co_title_col")
    co_desc_col = st.selectbox("Description Column (corrected file)", cols_corr, key="co_desc_col")

    co_forbidden_cols = [c for c in cols_corr if "predict" in c.lower()]
    co_selectable_cols = [c for c in cols_corr if c not in co_forbidden_cols]
    if not co_selectable_cols:
        st.error("❌ No valid **Correct Label** column available (columns containing 'predict' cannot be used).")
    else:
        co_corr_col = st.selectbox(
            "Correct Label Column (any name allowed)",
            co_selectable_cols,
            key="co_corr_col",
        )

        # Validate selection
        vals_norm_co = df_corr[co_corr_col].astype(str).str.strip().apply(normalize_sys_id)
        taxonomy_labels = set(folder_name_map.keys())
        co_match_tax = (vals_norm_co.isin(taxonomy_labels)).mean()
        co_match_trn = (vals_norm_co.isin(trained_labels)).mean()
        if co_match_tax < 0.2 and co_match_trn < 0.2:
            st.warning(
                f"⚠ Column '{co_corr_col}' has low match with taxonomy/trained labels "
                f"(taxonomy: {co_match_tax:.1%}, trained: {co_match_trn:.1%})."
            )

        if st.button("📤 Ingest Corrections", key="co_ingest_btn"):
            fb_rows = []
            for _, row in df_corr.iterrows():
                raw_corr = row.get(co_corr_col, "")
                if pd.isna(raw_corr) or str(raw_corr).strip() == "":
                    continue

                corr_norm = normalize_sys_id(str(raw_corr).strip())
                corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm

                pred_val = row.get("Predicted SysID", "")
                pred_norm = normalize_sys_id(str(pred_val)) if pred_val else ""
                if corr_norm == pred_norm:
                    continue

                fb_rows.append({
                    "Title": row.get(co_title_col, ""),
                    "Description": row.get(co_desc_col, ""),
                    "Predicted SysID": pred_val,  # may be empty if not provided by external file
                    "Correct SysID": corr_norm,   # ← user's selected column applied
                    "Model Version": meta.get("version", "unversioned"),
                    "Timestamp": datetime.utcnow().isoformat(),
                })

            if fb_rows:
                summary = append_feedback_rows(fb_rows)
                st.success(
                    f"Uploaded {summary['appended']} corrections. "
                    f"(Skipped blank: {summary['skipped_empty_correct']}, "
                    f"Skipped predicted==correct: {summary['skipped_equal_to_pred']})"
                )
            else:
                st.warning("⚠ No valid correction rows found in the corrected file.")

# ============================================================
# 🔍 Confidence Debugging & Model Introspection
# ============================================================
with st.expander("🔧 Confidence Debugger"):
    st.write("""
    This tool helps you verify whether your model version
    is producing **correct, calibrated confidence values**.
    
    Use it to compare unversioned vs versioned models, 
    test class mapping, and confirm calibration behavior.
    """)

    # Text input for manual debugging
    debug_input = st.text_input(
        "Enter sample text for debugging",
        value="lift outage on escalator - service interruption",
        key="debug_text_input"
    )

    if st.button("Run Debug Test", key="debug_run_btn"):

        try:
            # Vectorize
            Xd = vectorizer.transform([debug_input])

            # Predict
            raw_pred = model.predict(Xd)[0]
            pred_norm = normalize_sys_id(raw_pred)

            # Try canonical resolution
            canonical = resolve_hierarchical_sys_id(pred_norm, trained_labels)
            if canonical:
                pred_norm = canonical

            # Compute confidence
            from core.prediction import _confidence_from_model
            conf = _confidence_from_model(model, Xd, raw_pred)

            st.write("### 🔎 Debug Result")
            st.write(f"**Raw Predicted Label:** `{raw_pred}`")
            st.write(f"**Normalized Label:** `{pred_norm}`")
            st.write(f"**Confidence:** `{conf * 100:.2f}%`" if conf is not None else "**Confidence:** N/A")

            # Show class list and normalized class list
            raw_classes = getattr(model, "classes_", [])
            norm_classes = [normalize_sys_id(c) for c in raw_classes]

            st.write("### 🧬 Class Mapping")
            st.json({
                "raw_classes": raw_classes,
                "normalized_classes": norm_classes,
                "model_version": selected_version,
                "num_classes": len(raw_classes)
            })

        except Exception as e:
            st.error(f"Debug test failed: {e}")

    # ---------------------------------------------------------
    # Optional: canned samples for quick testing
    # ---------------------------------------------------------
    st.write("---")
    st.write("### 🧪 Quick Sample Tests")

    samples = [
        "safety inspection report for escalator machinery",
        "lift outage expected for maintenance",
        "cleaning schedule revision effective immediately",
        "fire safety procedure updated for platform 3",
        "escalator chain tension abnormal reading detected"
    ]

    if st.button("Run Sample Tests", key="debug_sample_btn"):

        results = []
        from core.prediction import _confidence_from_model

        for txt in samples:
            Xs = vectorizer.transform([txt])
            raw = model.predict(Xs)[0]
            norm = normalize_sys_id(raw)

            canonical = resolve_hierarchical_sys_id(norm, trained_labels)
            if canonical:
                norm = canonical

            conf = _confidence_from_model(model, Xs, raw)
            results.append({
                "text": txt,
                "raw_prediction": raw,
                "normalized_prediction": norm,
                "confidence": None if conf is None else f"{conf*100:.2f}%"
            })

        st.json(results)
st.caption(f"Model version: **{meta.get('version', selected_version)}**")