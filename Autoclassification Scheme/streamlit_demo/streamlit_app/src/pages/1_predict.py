# pages/Predict.py  —  v1.0 release
from datetime import datetime

import streamlit as st
import pandas as pd

from core.file_utils import load_csv_or_excel
from core.model_registry import list_model_versions, sync_from_mlflow
from core.model_loader import load_model_for_version, clear_model_caches
from core.prediction import (
    predict_single,
    predict_batch_df,
    _confidence_from_model,
    get_related_context,
)
from core.feedback import append_feedback_rows
from core.label_map import normalize_sys_id, resolve_hierarchical_sys_id
from core.config import CONFIG

st.set_page_config(page_title="Predict", page_icon="\U0001f52e", layout="wide")
st.title("\U0001f52e Document Folder Prediction")

# ============================================================
# Confidence Threshold Presets
# ============================================================
_PRESETS = {
    "Standard (calibrated)": {"low_upper": 0.23, "high_lower": 0.92},
    "Conservative (more suggestions)": {"low_upper": 0.35, "high_lower": 0.95},
    "Permissive (fewer suggestions)": {"low_upper": 0.15, "high_lower": 0.85},
}

with st.sidebar:
    st.subheader("\u2699\ufe0f Settings")
    preset_name = st.selectbox(
        "Confidence threshold",
        list(_PRESETS.keys()),
        index=0,
        key="conf_preset",
        help=(
            "Controls when neighbourhood suggestions appear. "
            "Conservative = more suggestions shown; "
            "Permissive = trust the model more."
        ),
    )
    active_bands = _PRESETS[preset_name]
    low_pct = int(active_bands["low_upper"] * 100)
    high_pct = int(active_bands["high_lower"] * 100)
    st.caption(f"LOW < {low_pct}% \u2022 MODERATE {low_pct}\u2013{high_pct}% \u2022 HIGH \u2265 {high_pct}%")

# ============================================================
# Load Versions
# ============================================================
versions = list_model_versions()
if not versions:
    st.error("\u274c No model versions found. Train a model first.")
    st.stop()

selected_version = st.selectbox(
    "\U0001f527 Select model version",
    options=versions,
    index=0,
    key="model_version_selector",
)

try:
    _, model, folder_name_map, trained_labels, meta = load_model_for_version(selected_version)
    st.session_state.pop("_sync_attempted", None)
except Exception as e:
    if not st.session_state.get("_sync_attempted"):
        st.session_state["_sync_attempted"] = True
        clear_model_caches()
        result = sync_from_mlflow()
        if result.get("purged"):
            st.toast(f"Removed stale versions: {', '.join(result['purged'])}")
        st.rerun()
    else:
        st.session_state.pop("_sync_attempted", None)
        st.error(f"\u274c Failed to load model '{selected_version}': {e}")
        st.stop()


# ============================================================
# Display helpers
# ============================================================
_REL_COLOURS = {
    "sibling": ("#e8f5e9", "#2e7d32"),
    "same branch": ("#e3f2fd", "#1565c0"),
    "same collection": ("#fff3e0", "#e65100"),
    "different collection": ("#fce4ec", "#b71c1c"),
}

def _rel_badge(relationship: str) -> str:
    bg, fg = _REL_COLOURS.get(relationship, ("#eeeeee", "#424242"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:10px;font-size:0.8em;font-weight:600;">'
        f'{relationship}</span>'
    )

_BAND_COLOURS = {
    "HIGH": ("#e8f5e9", "#2e7d32"),
    "MODERATE": ("#fff3e0", "#e65100"),
    "LOW": ("#fce4ec", "#c62828"),
}

def _band_badge(band: str) -> str:
    bg, fg = _BAND_COLOURS.get(band, ("#eeeeee", "#424242"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:10px;font-size:0.8em;font-weight:700;">'
        f'{band}</span>'
    )


def _fmt_sysid_name(sysid: str, name: str) -> str:
    """Format a SysID + folder name, avoiding duplication."""
    if name == sysid or not name:
        return f'<code>{sysid}</code>'
    return f'<code>{sysid}</code> \u2014 {name}'


def _render_related_context(ctx: dict, theme: str = "light") -> str:
    """Build HTML card for the Related Context section."""
    if not ctx or not ctx.get("prediction"):
        return ""

    band = ctx.get("confidence_band", "")
    related = ctx.get("related", [])
    siblings = ctx.get("siblings", [])
    collection = ctx.get("collection", {})
    parent = ctx.get("parent", {})
    hint = ctx.get("review_hint", "")
    joined = ctx.get("joined_suggestions", [])

    if theme == "dark":
        bg, fg, border = "#1e1e2e", "#e0e0e0", "#444"
        row_bg, row_alt = "#2a2a3d", "#23233a"
    else:
        bg, fg, border = "#f8f9fa", "#2a3f5f", "#dee2e6"
        row_bg, row_alt = "#ffffff", "#f1f3f5"

    band_html = _band_badge(band) if band else ""

    # --- Related folders table ---
    rows_html = ""
    for i, rel in enumerate(related):
        rbg = row_bg if i % 2 == 0 else row_alt
        badge = _rel_badge(rel["relationship"])
        conf_pct = f"{rel['confidence']*100:.1f}%"
        label = _fmt_sysid_name(rel['sysid'], rel['folder_name'])
        rows_html += (
            f'<tr style="background:{rbg};">'
            f'<td style="padding:6px 10px;">{label}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{badge}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:#888;">{conf_pct}</td>'
            f'</tr>'
        )

    # --- Sibling chips (only if NO joined suggestions; otherwise redundant) ---
    sib_html = ""
    if siblings and not joined:
        chips = "".join(
            f'<span style="background:#e8eaf6;color:#283593;padding:3px 10px;'
            f'border-radius:12px;margin:2px 4px;display:inline-block;font-size:0.82em;">'
            f'{s["sysid"]}</span>'
            for s in siblings[:6]
        )
        sib_html = (
            f'<div style="margin-top:12px;">'
            f'<b style="font-size:0.85em;">\U0001f4c1 Siblings under '
            f'{parent.get("name", "")}:</b><br>'
            f'<div style="margin-top:4px;">{chips}</div></div>'
        )

    # --- Joined-index neighbourhood suggestions (simplified) ---
    joined_html = ""
    if joined:
        items = ""
        for js in joined:
            par_label = _fmt_sysid_name(js['parent_sysid'], js['parent_name'])
            combined_pct = f"{js['combined_confidence']*100:.1f}%"
            # Children as compact list, showing SysID + name
            children_items = ""
            for c in js.get("children", [])[:3]:
                c_label = _fmt_sysid_name(c['sysid'], c['name'])
                c_pct = f"{c['confidence']*100:.1f}%"
                children_items += (
                    f'<div style="margin-left:18px;font-size:0.85em;color:#555;">'
                    f'\u2514 {c_label} ({c_pct})</div>'
                )
            items += (
                f'<div style="margin-bottom:10px;padding:8px;background:{row_bg};'
                f'border-radius:6px;border:1px solid {border};">'
                f'<b>{par_label}</b>'
                f'<span style="float:right;color:#1565c0;font-weight:600;">{combined_pct}</span>'
                f'{children_items}</div>'
            )

        joined_html = (
            f'<div style="margin-top:14px;padding-top:10px;border-top:1px solid {border};">'
            f'<b style="font-size:0.9em;">\U0001f517 Suggested Neighbourhoods</b>'
            f'<span style="font-size:0.8em;color:#888;"> \u2014 '
            f'parent folders where the correct class may live</span>'
            f'<div style="margin-top:8px;">{items}</div></div>'
        )

    html = f"""
    <div style="background:{bg};color:{fg};padding:18px;border-radius:10px;
                border:1px solid {border};margin-top:12px;">
        <h4 style="margin-top:0;">\U0001f9ed Related Context {band_html}</h4>
        <p style="font-size:0.9em;color:#666;margin-top:-8px;">{hint}</p>
        <div style="font-size:0.88em;margin-bottom:10px;">
            <b>Collection:</b> {collection.get('name', '')}
            &nbsp;\u2192&nbsp;
            <b>Parent:</b> {parent.get('name', '')}
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
            <tr style="background:{border};">
                <th style="padding:6px 10px;text-align:left;">Related Folder</th>
                <th style="padding:6px 10px;text-align:center;">Relationship</th>
                <th style="padding:6px 10px;text-align:right;">Confidence</th>
            </tr>
            {rows_html}
        </table>
        {sib_html}
        {joined_html}
    </div>
    """
    return html


# ============================================================
# 1) Single Prediction
# ============================================================
st.write("---")
st.header("\U0001f9c9 Single Prediction")

c1, c2 = st.columns([1, 2])

with c1:
    s_title = st.text_input("\U0001f4c4 Title", key="sp_title")
    s_desc = st.text_area("\U0001f4dd Description", height=120, key="sp_desc")

    if st.button("\U0001f52e Predict", key="sp_btn"):
        html, sysid, folder = predict_single(
            input_title=s_title,
            input_description=s_desc,
            model=model,
            folder_name_map=folder_name_map,
            valid_sysids=trained_labels,
            theme="light",
            confidence_bands=active_bands,
        )
        st.session_state["sp_html"] = html
        st.session_state["sp_sysid"] = sysid
        st.session_state["sp_folder"] = folder
        st.session_state["sp_title_v"] = s_title
        st.session_state["sp_desc_v"] = s_desc

        text = f"{s_title} {s_desc}".strip()
        ctx = get_related_context(
            model=model,
            text=text,
            folder_name_map=folder_name_map,
            trained_labels=trained_labels,
            top_n=6,
            confidence_bands=active_bands,
        )
        st.session_state["sp_context"] = ctx

with c2:
    if "sp_html" in st.session_state:
        st.markdown(st.session_state["sp_html"], unsafe_allow_html=True)

    if "sp_context" in st.session_state:
        ctx = st.session_state["sp_context"]
        ctx_html = _render_related_context(ctx, theme="light")
        if ctx_html:
            st.markdown(ctx_html, unsafe_allow_html=True)

# Correction block
if "sp_html" in st.session_state:
    st.subheader("\u270f\ufe0f Suggest a Correction")

    ctx = st.session_state.get("sp_context", {})
    correction_options = ["-- Type a custom SysID --"]
    if ctx:
        for rel in ctx.get("related", []):
            label = f"{rel['sysid']} \u2014 {rel['folder_name']} ({rel['relationship']})"
            correction_options.append(label)
        for sib in ctx.get("siblings", [])[:5]:
            label = f"{sib['sysid']} \u2014 {sib['folder_name']} (sibling)"
            if label not in correction_options:
                correction_options.append(label)
        for js in ctx.get("joined_suggestions", []):
            for child in js.get("children", [])[:3]:
                label = f"{child['sysid']} \u2014 {child['name']}"
                if label not in correction_options:
                    correction_options.append(label)

    selected_correction = st.selectbox(
        "Select from suggestions or type below:",
        options=correction_options,
        key="sp_corr_select",
    )

    if selected_correction == "-- Type a custom SysID --":
        corr_val = st.text_input("Correct SysID", key="sp_corr")
    else:
        corr_val = selected_correction.split(" \u2014 ")[0]
        parts = selected_correction.split(" \u2014 ", 1)
        desc = parts[1] if len(parts) > 1 else ""
        st.info(f"Selected: **{corr_val}** \u2014 {desc}")

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
            f"\u2713 Correction submitted "
            f"(duplicates: {summary['skipped_duplicate']}, "
            f"blank: {summary['skipped_empty_correct']}, "
            f"pred==corr: {summary['skipped_equal_to_pred']})"
        )

# ============================================================
# 2) Batch Prediction
# ============================================================
st.write("---")
st.header("\U0001f4e6 Batch Prediction")

pred_file = st.file_uploader(
    "Upload CSV/XLSX (Title, Description, optional Correct Label)",
    type=["csv", "xlsx", "xls"],
    key="bp_file",
)

if pred_file:
    df_pred = load_csv_or_excel(pred_file)
    st.subheader("\U0001f4c4 File Preview")
    st.dataframe(df_pred.head(20), use_container_width=True)

    cols_pred = list(df_pred.columns)

    st.subheader("\U0001fa84 Step A \u2014 Select Columns")
    bp_title_col = st.selectbox("Title Column", cols_pred, key="bp_title_col")
    bp_desc_col = st.selectbox("Description Column", cols_pred, key="bp_desc_col")

    forbidden = [c for c in cols_pred if "predict" in c.lower()]
    selectable_corr = [c for c in cols_pred if c not in forbidden]
    bp_corr_col = None
    if selectable_corr:
        bp_corr_col = st.selectbox(
            "Correct Label Column (optional)",
            selectable_corr,
            key="bp_corr_col",
        )
    else:
        st.warning("No valid Correct Label column found.")

    st.subheader("\U0001fa84 Step B \u2014 Run Prediction")
    if st.button("\U0001f52e Predict Entire File", key="bp_predict_btn"):
        preds = predict_batch_df(
            df=df_pred,
            title_col=bp_title_col,
            desc_col=bp_desc_col,
            model=model,
            folder_name_map=folder_name_map,
            valid_sysids=trained_labels,
            true_sysid_col=None,
            confidence_bands=active_bands,
        )
        preds["Correct SysID"] = df_pred[bp_corr_col].astype(str) if bp_corr_col else ""
        st.session_state["bp_preds"] = preds
        st.success(f"Predicted {len(preds)} rows.")

    if isinstance(st.session_state.get("bp_preds"), pd.DataFrame):
        preds = st.session_state["bp_preds"]
        st.subheader("\U0001f4ca Predictions")

        display_cols = [
            "Title", "Predicted SysID", "Predicted Folder",
            "Confidence", "Band", "Related Context",
        ]
        if "Correct SysID" in preds.columns:
            display_cols.append("Correct SysID")
        available = [c for c in display_cols if c in preds.columns]
        st.dataframe(preds[available].head(25), use_container_width=True)

        with st.expander("Show all columns"):
            st.dataframe(preds.head(25), use_container_width=True)

        downloadable = preds.copy()
        if "Correct SysID" not in downloadable.columns:
            downloadable["Correct SysID"] = ""
        st.download_button(
            "\u2b07\ufe0f Download predictions",
            downloadable.to_csv(index=False).encode("utf-8"),
            "predictions_for_correction.csv",
            "text/csv",
            key="bp_download_template",
        )

    st.subheader("\U0001fa84 Step C \u2014 Ingest Corrections")
    if bp_corr_col is None:
        st.info("Select a Correct Label Column above to ingest corrections.")
    else:
        if st.button("\U0001f4e4 Ingest Corrections From This File", key="bp_ingest_btn"):
            fb_rows = []
            preds_df = st.session_state.get("bp_preds")

            if isinstance(preds_df, pd.DataFrame) and "Source Index" in preds_df.columns:
                for _, prow in preds_df.iterrows():
                    idx = prow.get("Source Index")
                    if idx not in df_pred.index:
                        continue
                    raw_corr = df_pred.at[idx, bp_corr_col]
                    if pd.isna(raw_corr) or str(raw_corr).strip() == "":
                        continue
                    corr_norm = normalize_sys_id(str(raw_corr).strip())
                    corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm
                    fb_rows.append({
                        "Title": prow.get("Title", ""),
                        "Description": prow.get("Description", ""),
                        "Predicted SysID": prow.get("Predicted SysID", ""),
                        "Correct SysID": corr_norm,
                        "Model Version": meta.get("version"),
                        "Timestamp": datetime.utcnow().isoformat(),
                    })
            else:
                for _, row in df_pred.iterrows():
                    raw_corr = row.get(bp_corr_col, "")
                    if pd.isna(raw_corr) or str(raw_corr).strip() == "":
                        continue
                    corr_norm = normalize_sys_id(str(raw_corr).strip())
                    corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm
                    fb_rows.append({
                        "Title": row.get(bp_title_col, ""),
                        "Description": row.get(bp_desc_col, ""),
                        "Predicted SysID": row.get("Predicted SysID", ""),
                        "Correct SysID": corr_norm,
                        "Model Version": meta.get("version"),
                        "Timestamp": datetime.utcnow().isoformat(),
                    })

            if fb_rows:
                summary = append_feedback_rows(fb_rows)
                st.success(
                    f"Uploaded {summary['appended']} corrections "
                    f"(duplicates: {summary['skipped_duplicate']}, "
                    f"blank: {summary['skipped_empty_correct']}, "
                    f"pred==corr: {summary['skipped_equal_to_pred']})"
                )
            else:
                st.warning("\u26a0 No valid correction rows found.")

# ============================================================
# 3) Corrections Only
# ============================================================
st.write("---")
st.header("\u2705 Corrections Only")

corr_file = st.file_uploader(
    "Upload corrected CSV/XLSX (Title, Description, Correct Label)",
    type=["csv", "xlsx", "xls"],
    key="co_file",
)

if corr_file:
    df_corr = load_csv_or_excel(corr_file)
    st.dataframe(df_corr.head(20), use_container_width=True)

    cols_corr = list(df_corr.columns)
    co_title_col = st.selectbox("Title Column", cols_corr, key="co_title_col")
    co_desc_col = st.selectbox("Description Column", cols_corr, key="co_desc_col")

    forbidden_corr = [c for c in cols_corr if "predict" in c.lower()]
    selectable_corr_cols = [c for c in cols_corr if c not in forbidden_corr]
    co_corr_col = None
    if selectable_corr_cols:
        co_corr_col = st.selectbox("Correct Label Column", selectable_corr_cols, key="co_corr_col")
    else:
        st.error("\u274c No valid Correct Label column found.")

    if co_corr_col:
        if st.button("\U0001f4e4 Ingest Corrections", key="co_ingest_btn"):
            fb_rows = []
            for _, row in df_corr.iterrows():
                raw_corr = row.get(co_corr_col, "")
                if pd.isna(raw_corr) or str(raw_corr).strip() == "":
                    continue
                corr_norm = normalize_sys_id(str(raw_corr).strip())
                corr_norm = resolve_hierarchical_sys_id(corr_norm, trained_labels) or corr_norm
                fb_rows.append({
                    "Title": row.get(co_title_col, ""),
                    "Description": row.get(co_desc_col, ""),
                    "Predicted SysID": row.get("Predicted SysID", ""),
                    "Correct SysID": corr_norm,
                    "Model Version": meta.get("version"),
                    "Timestamp": datetime.utcnow().isoformat(),
                })
            if fb_rows:
                summary = append_feedback_rows(fb_rows)
                st.success(
                    f"Uploaded {summary['appended']} corrections "
                    f"(duplicates: {summary['skipped_duplicate']}, "
                    f"blank: {summary['skipped_empty_correct']}, "
                    f"pred==corr: {summary['skipped_equal_to_pred']})"
                )
            else:
                st.warning("\u26a0 No valid corrections found.")

# ============================================================
# Developer Tools (collapsed by default)
# ============================================================
st.write("---")
with st.expander("\U0001f527 Developer Tools", expanded=False):
    st.subheader("Confidence Debugger")
    dbg_text = st.text_input(
        "Enter text for debugging",
        value="lift outage on escalator machinery",
        key="debug_text_input",
    )
    if st.button("Run Debug Test", key="debug_btn"):
        try:
            dbg_df = pd.DataFrame({"text": [dbg_text]})
            pred = model.predict(dbg_df)[0]
            norm = normalize_sys_id(pred)
            sm_norm = resolve_hierarchical_sys_id(norm, trained_labels) or norm
            confidence = _confidence_from_model(model, dbg_df, pred)
            conf_str = f"{confidence * 100:.2f}%" if isinstance(confidence, float) else "N/A"
            st.write(f"**Raw Pred:** `{pred}`")
            st.write(f"**Normalized:** `{sm_norm}`")
            st.write(f"**Confidence:** `{conf_str}`")
        except Exception as e:
            st.error(f"Debug failed: {e}")

st.caption(f"Model version: **{meta.get('version', selected_version)}** \u2022 Threshold: {preset_name}")
