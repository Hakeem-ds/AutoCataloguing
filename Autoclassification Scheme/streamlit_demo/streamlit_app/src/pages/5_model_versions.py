# pages/5_model_versions.py
import io
import pickle
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import mlflow

from core.model_registry import (
    list_model_versions,
    load_model_by_version,
    get_model_registry,
    get_promoted_version,
    set_promoted_version,
    sync_from_mlflow,
)
from core.model_loader import load_current_model, load_model_for_version
from core.file_utils import safe_rerun
from core.config import CONFIG
from core.label_map import normalize_sys_id


st.set_page_config(page_title="Model Versions", page_icon="\U0001f9ec", layout="wide")
st.title("\U0001f9ec Model Versions")

with st.expander("\U0001f4d8 Page Overview", expanded=False):
    st.markdown("""
This page displays **all registered model versions**:
- \u2b50 **Promote** a version to become the *current* model
- \U0001f4ca Compare **metrics** across versions (accuracy, F1, precision, recall)
- \U0001f4c8 Track **improvement** with delta indicators
- \U0001f9e9 Inspect **model internals** (TF-IDF, calibration, size)
- \U0001f9ea **Smoke test** any version
- \U0001fa65 Generate a **Model Card**
""")


# ==================================================
# Helpers
# ==================================================
def _extract_run_id_from_uri(uri: str):
    if not uri or not uri.startswith("runs:"):
        return None
    try:
        return uri.split("/")[1]
    except Exception:
        return None


def _fetch_run(run_id: str):
    try:
        client = mlflow.tracking.MlflowClient()
        return client.get_run(run_id)
    except Exception:
        return None


def _metrics_and_tags(run_id: str):
    run = _fetch_run(run_id)
    if not run:
        return {}, {}
    return dict(run.data.metrics or {}), dict(run.data.tags or {})


def _safe_dt(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
    except Exception:
        return None


def _human_bytes(n):
    if n is None:
        return "N/A"
    x = float(n)
    for u in ["B", "KB", "MB", "GB"]:
        if x < 1024:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{x:.1f} TB"


def _fmt_pct(val):
    if val is None:
        return "\u2014"
    try:
        return f"{float(val) * 100:.1f}%"
    except Exception:
        return "\u2014"


def _fmt_num(val, decimals=3):
    if val is None:
        return "\u2014"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return "\u2014"


def _delta_str(current, previous):
    if current is None or previous is None:
        return None
    try:
        d = float(current) - float(previous)
        return f"{d:+.4f}"
    except Exception:
        return None


def _extract_vectorizer_stats(model):
    stats = {
        "vocab_size": None, "ngram_range": None, "min_df": None,
        "max_df": None, "sublinear_tf": None, "norm": None, "use_idf": None,
    }
    try:
        steps = getattr(model, "named_steps", {})
        vec = steps.get("vectorizer")
        if vec is None:
            return stats
        if hasattr(vec, "get_feature_names_out"):
            stats["vocab_size"] = len(vec.get_feature_names_out())
        elif hasattr(vec, "vocabulary_"):
            stats["vocab_size"] = len(vec.vocabulary_)
        p = vec.get_params()
        for k in ["ngram_range", "min_df", "max_df", "sublinear_tf", "norm", "use_idf"]:
            stats[k] = p.get(k)
        return stats
    except Exception:
        return stats


def _model_size_bytes(model):
    try:
        return len(pickle.dumps(model))
    except Exception:
        return None


# ==================================================
# Sync from MLflow
# ==================================================
if st.button("\U0001f504 Sync from MLflow", help="Discover new models from MLflow experiment"):
    with st.spinner("Scanning MLflow experiment..."):
        result = sync_from_mlflow()
    if result["added"]:
        st.success(f"Found {len(result['added'])} new model(s)")
        safe_rerun()
    elif result["errors"]:
        st.error(f"Errors: {result['errors']}")
    else:
        st.info("All MLflow models already registered.")

# ==================================================
# Load Versions
# ==================================================
versions = list_model_versions()
if not versions:
    st.error("\u274c No models found. Train one first.")
    st.stop()

promoted = get_promoted_version()
registry = get_model_registry()

rows = []
for v in versions:
    entry = registry.get(v, {})
    if not isinstance(entry, dict) or v == "__meta__":
        continue
    uri = entry.get("mlflow_model_uri")
    run_id = _extract_run_id_from_uri(uri) if uri else None
    row_data = {
        "version": v, "promoted": ("\u2b50" if v == promoted else ""),
        "trained_on": entry.get("trained_on", ""),
        "pipeline": entry.get("pipeline_version", ""),
        "uri": uri or "",
    }
    if run_id:
        metrics, _ = _metrics_and_tags(run_id)
        for key in ["accuracy", "macro_f1", "weighted_f1",
                     "macro_precision", "weighted_precision",
                     "n_training_samples", "n_classes",
                     "mean_confidence", "predict_time_sec", "fit_time_sec"]:
            row_data[key] = metrics.get(key)
    rows.append(row_data)

df = pd.DataFrame(rows)

def _rank(row):
    promoted_rank = 0 if row["promoted"] else 1
    ts = _safe_dt(row["trained_on"]) or datetime(1970, 1, 1, tzinfo=timezone.utc)
    acc_val = -1.0 if row.get("accuracy") is None else float(row["accuracy"])
    return (promoted_rank, -ts.timestamp(), -acc_val)

df["_rank"] = df.apply(_rank, axis=1)
df = df.sort_values("_rank").drop(columns=["_rank"])

# ==================================================
# Overview Table (compact)
# ==================================================
st.subheader("\U0001f4ca All Versions")

df_show = df.copy()
for col in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "weighted_precision", "mean_confidence"]:
    if col in df_show.columns:
        df_show[col] = df_show[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014")
for col in ["predict_time_sec", "fit_time_sec"]:
    if col in df_show.columns:
        df_show[col] = df_show[col].apply(lambda x: f"{x:.2f}s" if pd.notna(x) else "\u2014")
for col in ["n_training_samples", "n_classes"]:
    if col in df_show.columns:
        df_show[col] = df_show[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "\u2014")

show_cols = [c for c in [
    "promoted", "version", "trained_on", "pipeline",
    "n_training_samples", "accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "mean_confidence",
] if c in df_show.columns]

st.dataframe(df_show[show_cols], use_container_width=True, hide_index=True,
             height=min(400, 38 + 35 * len(df_show)))

# Comparison chart
if len(df) > 1:
    chart_cols = [c for c in ["accuracy", "macro_f1", "weighted_f1", "macro_precision"] if c in df.columns]
    if chart_cols:
        chart_df = df[["version"] + chart_cols].set_index("version").dropna(how="all")
        if not chart_df.empty:
            with st.expander("\U0001f4c8 Version Comparison Chart", expanded=False):
                st.bar_chart(chart_df, height=250)

st.divider()

# ==================================================
# Inspect a Selected Version
# ==================================================
col_sel, col_promote = st.columns([3, 1])
with col_sel:
    choice = st.selectbox("Select version to inspect", list(df["version"]), index=0)
with col_promote:
    st.write("")  # spacer
    st.write("")
    if st.button("\u2b50 Promote this version"):
        try:
            set_promoted_version(choice)
            st.success(f"Promoted '{choice}'")
            safe_rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

# Load the selected model
_, mdl_sel, fmap_sel, labels_sel, meta_sel = load_model_for_version(choice)
uri_sel = meta_sel.get("mlflow_model_uri")
run_id_sel = _extract_run_id_from_uri(uri_sel) if uri_sel else None
metrics_sel, tags_sel = ({}, {})
if run_id_sel:
    metrics_sel, tags_sel = _metrics_and_tags(run_id_sel)

# Previous model metrics (from tags)
prev = {}
for key in ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
            "weighted_precision", "mean_confidence", "n_training_samples"]:
    tag_val = tags_sel.get(f"prev_{key}")
    if tag_val:
        try:
            prev[key] = float(tag_val)
        except Exception:
            pass

# ==================================================
# Tabbed detail view
# ==================================================
tab_perf, tab_detail, tab_test, tab_card = st.tabs(
    ["\U0001f4c8 Performance", "\U0001f9e9 Model Details", "\U0001f9ea Smoke Test", "\U0001fa65 Model Card"]
)

# ── Tab: Performance ──
with tab_perf:
    # Key metrics in a compact 2-column table
    st.markdown(f"**Version:** `{choice}`  \u2022  **Trained:** {meta_sel.get('trained_on', 'N/A')}  \u2022  **Pipeline:** {meta_sel.get('pipeline_version', 'N/A')}")

    perf_left, perf_right = st.columns(2)

    with perf_left:
        st.markdown("##### Core Metrics")
        perf_data = {
            "Metric": ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Weighted Precision",
                       "Macro Recall", "Weighted Recall"],
            "Value": [
                _fmt_pct(metrics_sel.get("accuracy")),
                _fmt_pct(metrics_sel.get("macro_f1")),
                _fmt_pct(metrics_sel.get("weighted_f1")),
                _fmt_pct(metrics_sel.get("macro_precision")),
                _fmt_pct(metrics_sel.get("weighted_precision")),
                _fmt_pct(metrics_sel.get("macro_recall")),
                _fmt_pct(metrics_sel.get("weighted_recall")),
            ],
            "vs Previous": [
                _delta_str(metrics_sel.get("accuracy"), prev.get("accuracy")) or "",
                _delta_str(metrics_sel.get("macro_f1"), prev.get("macro_f1")) or "",
                _delta_str(metrics_sel.get("weighted_f1"), prev.get("weighted_f1")) or "",
                _delta_str(metrics_sel.get("macro_precision"), prev.get("macro_precision")) or "",
                _delta_str(metrics_sel.get("weighted_precision"), prev.get("weighted_precision")) or "",
                "", "",
            ],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    with perf_right:
        st.markdown("##### Training & Confidence")
        n_samples = metrics_sel.get("n_training_samples")
        n_classes = metrics_sel.get("n_classes")
        info_data = {
            "": ["Training Samples", "Classes", "Mean Confidence", "Median Confidence",
                 "Confidence Source", "Predict Time", "Fit Time"],
            "Value": [
                f"{int(n_samples):,}" if n_samples else str(len(labels_sel)) + " (from model)",
                f"{int(n_classes):,}" if n_classes else str(len(labels_sel)),
                _fmt_pct(metrics_sel.get("mean_confidence")),
                _fmt_pct(metrics_sel.get("median_confidence")),
                tags_sel.get("confidence_source", "\u2014"),
                f"{metrics_sel.get('predict_time_sec', 0):.3f}s" if metrics_sel.get("predict_time_sec") else "\u2014",
                f"{metrics_sel.get('fit_time_sec', 0):.3f}s" if metrics_sel.get("fit_time_sec") else "\u2014",
            ],
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

# ── Tab: Model Details ──
with tab_detail:
    det_left, det_right = st.columns(2)

    vec_stats = _extract_vectorizer_stats(mdl_sel)
    mdl_size = _model_size_bytes(mdl_sel)

    with det_left:
        st.markdown("##### TF-IDF Vectorizer")
        vec_data = {
            "Setting": ["Vocabulary Size", "n-gram Range", "min_df", "max_df", "sublinear_tf", "norm", "use_idf"],
            "Value": [
                f"{vec_stats['vocab_size']:,}" if vec_stats["vocab_size"] else "\u2014",
                str(vec_stats["ngram_range"]),
                str(vec_stats["min_df"]),
                str(vec_stats["max_df"]),
                str(vec_stats["sublinear_tf"]),
                str(vec_stats["norm"]),
                str(vec_stats["use_idf"]),
            ],
        }
        st.dataframe(pd.DataFrame(vec_data), use_container_width=True, hide_index=True)

    with det_right:
        st.markdown("##### Classifier & Calibration")
        cal_data = {
            "Property": ["Classifier Type", "Model Size", "Classes", "MLflow URI",
                         "Calibration Requested", "Calibration Used", "Calibration Method",
                         "Effective CV Folds", "Min Class Count"],
            "Value": [
                type(mdl_sel.named_steps.get("classifier", mdl_sel)).__name__,
                _human_bytes(mdl_size),
                str(len(labels_sel)),
                uri_sel or "\u2014",
                tags_sel.get("calibration_requested", "\u2014"),
                tags_sel.get("calibration_used", "\u2014"),
                tags_sel.get("calibration_method", tags_sel.get("calibration_decision_reason", "\u2014")),
                tags_sel.get("effective_cv_folds", "\u2014"),
                tags_sel.get("min_class_count", "\u2014"),
            ],
        }
        st.dataframe(pd.DataFrame(cal_data), use_container_width=True, hide_index=True)

# ── Tab: Smoke Test ──
with tab_test:
    smoke_text = st.text_input("Enter sample text", "escalator maintenance and lift outage")
    if st.button("Run Smoke Test"):
        try:
            raw = mdl_sel.predict(pd.DataFrame({"text": [smoke_text]}))[0]
            norm = normalize_sys_id(raw)
            st.success(f"Prediction: `{raw}` \u2192 Normalized: `{norm}`")
        except Exception as e:
            st.error(f"Smoke test failed: {e}")

# ── Tab: Model Card ──
with tab_card:
    if st.button("Generate Model Card"):
        buf = io.StringIO()
        buf.write(f"# Model Card \u2014 {choice}\n\n")
        buf.write(f"- **Version:** {meta_sel.get('version')}\n")
        buf.write(f"- **Trained On:** {meta_sel.get('trained_on')}\n")
        buf.write(f"- **Pipeline Version:** {meta_sel.get('pipeline_version')}\n")
        buf.write(f"- **MLflow URI:** `{uri_sel}`\n")
        buf.write(f"- **Promoted:** {'Yes' if choice == promoted else 'No'}\n")
        buf.write(f"\n## Training Data\n")
        buf.write(f"- Samples: {int(n_samples):,}\n" if n_samples else "- Samples: N/A\n")
        buf.write(f"- Classes: {int(n_classes):,}\n" if n_classes else f"- Classes: {len(labels_sel)}\n")
        buf.write(f"\n## Performance\n")
        for m in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "weighted_precision", "macro_recall", "weighted_recall"]:
            buf.write(f"- {m}: {_fmt_pct(metrics_sel.get(m))}\n")
        buf.write(f"- Mean Confidence: {_fmt_pct(metrics_sel.get('mean_confidence'))}\n")
        buf.write(f"\n## Model Stats\n")
        buf.write(f"- Vocab Size: {vec_stats.get('vocab_size')}\n")
        buf.write(f"- n-gram Range: {vec_stats.get('ngram_range')}\n")
        buf.write(f"- Serialised Size: {_human_bytes(mdl_size)}\n")
        buf.write(f"\n---\n_Generated {datetime.utcnow().isoformat()}Z_\n")
        st.download_button("\u2b07\ufe0f Download Model Card", buf.getvalue().encode(), f"model_card_{choice}.md", "text/markdown")
