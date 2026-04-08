# pages/Review_and_Retrain.py
import os
import pandas as pd
import streamlit as st

from core.file_utils import load_csv_or_excel
from core.config import CONFIG
from core.feedback import append_feedback_rows, load_feedback, deduplicate_feedback
from core.training_data import (
    build_training_from_feedback,
    append_training_rows,
    run_retraining_mlflow_via_job,
    clear_feedback_and_training,
)
from core.model_loader import clear_model_caches
from core.training_metadata import (
    get_retrain_status,
    get_metadata,
    peek_next_version_name,
)


st.set_page_config(page_title="Review & Retrain", page_icon="\U0001f501", layout="wide")
st.title("\U0001f501 Review & Retrain")

with st.expander("\U0001f4d8 Flow overview", expanded=True):
    st.markdown("""
This page enables a **clean retraining loop** driven by **user feedback**.

**End-to-end loop:**
1) **Review/Add Feedback** \u2014 Upload or add corrections (deduplicated)  
2) **Build Training Data** \u2014 Convert feedback into `training_data.csv`  
3) **Retrain (MLflow Job)** \u2014 Streamlit queues the CSV as an MLflow artifact and triggers a **Databricks Job**  
4) **Register Version** \u2014 The job logs a **model manifest**, Streamlit updates the registry with the new **MLflow model URI**  

**Notes:**
- We **include singleton classes** (labels with only 1 row) in **training** so the model can **learn as it goes**.  
- We display **singleton warnings** so you can decide whether to proceed or add more data first.
    """)

# =====================================================================
# Helpers
# =====================================================================

def _safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for safe display.
    Resets index and normalises all values to clean strings.
    """
    out = df.copy().reset_index(drop=True)
    for col in out.columns:
        out[col] = out[col].astype(str).replace({"nan": "", "None": "", "NaT": ""})
    return out


def _load_training_csv() -> pd.DataFrame:
    """Load artifacts/training_data.csv if present; else empty DataFrame."""
    path = CONFIG["training_data_csv"]
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=["Title", "Description", "SysID", "RowID"])
    return pd.DataFrame(columns=["Title", "Description", "SysID", "RowID"])


def _profile_training_data(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "samples": 0,
            "labels": 0,
            "singletons_count": 0,
            "singletons_ratio": 0.0,
            "counts_df": pd.DataFrame(columns=["SysID", "count"]).astype({"count": "int"}),
        }

    counts = df["SysID"].astype(str).value_counts().rename_axis("SysID").reset_index(name="count")
    samples = int(len(df))
    labels = int(len(counts))
    singletons = counts[counts["count"] == 1]
    singletons_count = int(len(singletons))
    singletons_ratio = (singletons_count / labels) if labels > 0 else 0.0

    return {
        "samples": samples,
        "labels": labels,
        "singletons_count": singletons_count,
        "singletons_ratio": singletons_ratio,
        "counts_df": counts,
    }


def _singleton_advice(singletons_count: int, samples: int) -> str:
    if singletons_count == 0:
        return "\u2705 No singleton classes detected."
    if samples <= 100:
        return "Consider collecting a bit more data for singleton classes to improve reliability."
    return "Proceeding is okay \u2014 model will learn from singletons, but more examples will help."


# =====================================================================
# Step 1 \u2014 Feedback
# =====================================================================

st.header("1) \U0001f4dd Review / Add Feedback")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Feedback")
    fb = load_feedback()
    if fb.empty:
        st.info("No feedback yet.")
    else:
        st.write(f"Total feedback rows: **{len(fb)}**")
        # Truncate long text for display
        fb_display = fb.tail(25).copy()
        for col in ["Title", "Description"]:
            if col in fb_display.columns:
                fb_display[col] = fb_display[col].astype(str).str[:80]
        # Use st.table() — AG Grid (st.dataframe) triggers React error #185
        # on this page due to a known Streamlit frontend re-render bug.
        st.table(_safe_display_df(fb_display))

        # Dedup button
        if st.button("\U0001f9f9 Deduplicate Feedback", key="dedup_fb_btn"):
            result = deduplicate_feedback()
            if result["removed"] > 0:
                st.success(
                    f"Removed **{result['removed']}** duplicates. "
                    f"Feedback: {result['before']} \u2192 {result['after']} rows."
                )
            else:
                st.info("No duplicates found \u2014 feedback is already clean.")

with col2:
    st.subheader("Add Feedback (CSV)")
    st.markdown("CSV columns expected: **Title**, **Description**, **Predicted SysID**, **Correct SysID** (optional)")
    up = st.file_uploader("Upload Feedback CSV", type=["csv", "xlsx", "xls"], key="fb_uploader")

    if up is not None:
        try:
            df_up = load_csv_or_excel(up)
            st.table(_safe_display_df(df_up.head(10)))
            st.caption(f"{len(df_up)} rows detected in upload.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_up = None

        if df_up is not None and st.button("\U0001f4e4 Ingest Feedback", key="fb_ingest_btn"):
            try:
                rows = df_up.to_dict(orient="records")
                summary = append_feedback_rows(rows)
                total_now = len(load_feedback())
                st.success(
                    f"Ingested **{summary['appended']}** new rows. "
                    f"Total feedback: **{total_now}**. "
                    f"(duplicates skipped: {summary['skipped_duplicate']}, "
                    f"blank: {summary['skipped_empty_correct']}, "
                    f"pred==corr: {summary['skipped_equal_to_pred']})"
                )
            except Exception as e:
                st.error(f"Failed to append feedback: {e}")

st.write("---")

# =====================================================================
# Step 2 \u2014 Training set
# =====================================================================

st.header("2) \U0001f9f0 Build Training Data from Feedback")

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
            st.success(
                f"Training data built with **{len(train_df)}** rows "
                f"(duplicates automatically removed)."
            )
            st.session_state["last_train_preview"] = train_df.head(200)

    td = _load_training_csv()
    prof = _profile_training_data(td)

    st.subheader("\U0001f4ca Training Dataset Profile")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", f"{prof['samples']}")
    m2.metric("Labels", f"{prof['labels']}")
    m3.metric("Singleton classes", f"{prof['singletons_count']}")
    m4.metric("Singleton ratio", f"{prof['singletons_ratio']*100:.1f}%")

    if prof["singletons_count"] > 0:
        st.warning(
            f"\u26a0\ufe0f Detected **{prof['singletons_count']}** singleton label(s). "
            f"{_singleton_advice(prof['singletons_count'], prof['samples'])}"
        )
        st.caption("Singletons are included in **training only**; they will not be placed in test. This is intentional.")

    with st.expander("\U0001f50e View label counts (top 200)", expanded=False):
        if prof["counts_df"].empty:
            st.info("No training data yet.")
        else:
            # Keep st.dataframe for label counts (200 rows needs scrolling)
            st.dataframe(
                _safe_display_df(prof["counts_df"].head(200)),
                use_container_width=True,
                height=300,
                hide_index=True,
            )

    with st.expander("\U0001f4c8 Quick chart (top 20 labels)", expanded=False):
        if not prof["counts_df"].empty:
            chart_df = prof["counts_df"].head(20).set_index("SysID")
            st.bar_chart(chart_df)

    if not td.empty:
        st.download_button(
            "\u2b07\ufe0f Download training_data.csv",
            data=td.to_csv(index=False).encode("utf-8"),
            file_name="training_data.csv",
            mime="text/csv",
        )

with c2:
    st.subheader("Add external training rows (optional)")
    st.markdown("CSV columns required: **Title**, **Description**, **SysID**")
    ext = st.file_uploader("Upload extra training CSV", type=["csv", "xlsx", "xls"], key="train_uploader")

    if ext is not None:
        try:
            tdf = load_csv_or_excel(ext)
            st.table(_safe_display_df(tdf.head(10)))
            st.caption(f"{len(tdf)} rows detected in upload.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            tdf = None

        if tdf is not None and st.button("\U0001f4e4 Ingest Training Rows", key="train_ingest_btn"):
            try:
                rows = tdf.to_dict(orient="records")
                merged = append_training_rows(rows)
                st.success(f"Appended. Training set now has **{len(merged)}** rows (deduplicated).")
            except Exception as e:
                st.error(f"Failed to append training rows: {e}")

st.write("---")

# =====================================================================
# Retrain Advisory Dashboard
# =====================================================================

st.header("\U0001f4ca Retrain Advisory")

try:
    status = get_retrain_status()

    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
    adv_col1.metric("New Corrections", f"{status['new_corrections']}")
    adv_col2.metric("Threshold", f"{status['threshold']}")
    adv_col3.metric("Retrains Completed", f"{status['retrains_completed']}")
    adv_col4.metric("Next Version", status["next_version"])

    pct = status["progress_pct"]
    bar_text = f"{status['new_corrections']}/{status['threshold']} corrections ({pct}%)"
    st.progress(pct / 100, text=bar_text)

    if status["is_locked"]:
        st.warning("\U0001f512 A retrain is currently in progress. Please wait for it to complete.")
    elif status["should_retrain"]:
        st.success(
            f"\u2705 **Ready to retrain!** {status['new_corrections']} new corrections "
            f"meet the threshold of {status['threshold']}."
        )
    else:
        remaining = status["threshold"] - status["new_corrections"]
        st.info(f"\U0001f4dd Collect **{remaining}** more corrections before next retrain is recommended.")

    if status["last_retrain_at"]:
        st.caption(f"Last retrain: {status['last_retrain_at']}")

    with st.expander("\U0001f4dc Retrain History", expanded=False):
        meta = get_metadata()
        retrains = meta.get("retrains", [])
        if not retrains:
            st.info("No retrains logged yet.")
        else:
            hist_df = pd.DataFrame(retrains)
            display_cols = [c for c in [
                "retrain_id", "version", "timestamp", "dataset_size",
                "new_corrections", "accuracy", "promoted", "threshold_used", "notes"
            ] if c in hist_df.columns]
            st.table(_safe_display_df(hist_df[display_cols]))

except Exception as e:
    st.warning(f"Could not load retrain advisory: {e}")

st.write("---")

# =====================================================================
# Step 3 \u2014 Retrain Model (MLflow Job)
# =====================================================================

st.header("3) \U0001f916 Retrain Model")

st.markdown("""
This will:

1) Log the current **training_data.csv** as a **queued MLflow artifact**  
2) Queue a job to run `main.py --queue_run_id <id> --queue_artifact_path queued/training_data.csv --previous_model_uri runs:/<promoted>/model` 
3) When the job finishes, Streamlit pulls the **model manifest** and registers a new model version
""")

if _load_training_csv().empty:
    st.info("No training_data.csv present yet. Build it in Step 2.")
else:
    st.caption("We include singleton classes in **training** to ensure new labels are learned early.")

if st.button("Start retraining", key="retrain_btn", type="primary"):

    progress_bar = st.progress(0, text="Preparing...")
    status_container = st.status("\U0001f504 Retraining in progress...", expanded=True)

    _step_labels = {
        1: "Queueing training data as MLflow artifact...",
        2: "Triggering Databricks training job...",
        3: "Waiting for training job to complete...",
        4: "Registering new model version...",
        5: "Retraining complete!",
    }
    _step_progress = {1: 0.10, 2: 0.20, 3: 0.40, 4: 0.85, 5: 1.0}

    def _on_step(name, step_num, total):
        pct = _step_progress.get(step_num, step_num / total)
        label = _step_labels.get(step_num, name)
        progress_bar.progress(pct, text=label)
        status_container.write(f"**Step {step_num}/{total}:** {label}")

    def _on_poll(life_cycle_state, elapsed_secs):
        import math
        fraction = 1.0 - math.exp(-elapsed_secs / 300.0)
        pct = 0.40 + fraction * 0.45
        mins = int(elapsed_secs // 60)
        secs = int(elapsed_secs % 60)
        progress_bar.progress(
            min(pct, 0.84),
            text=f"Job {life_cycle_state.lower()}... ({mins}m {secs}s elapsed)"
        )

    try:
        results = run_retraining_mlflow_via_job(
            on_step=_on_step,
            on_poll=_on_poll,
        )

        progress_bar.progress(1.0, text="\u2705 Retraining complete!")
        status_container.update(label="\u2705 Retraining complete!", state="complete", expanded=True)
        clear_model_caches()

        if results.get("promoted"):
            status_container.write(f"\U0001f3c6 **Auto-promoted** \u2014 {results.get('promote_reason', '')}")
        else:
            status_container.write(f"\U0001f4cb **Kept as candidate** \u2014 {results.get('promote_reason', 'Manual review needed')}")

        status_container.write(
            f"**Version:** {results['version']}\n\n"
            f"**Accuracy:** {results.get('accuracy', 'N/A')}\n\n"
            f"**Model URI:** `{results['model_uri']}`\n\n"
            f"**Job Run ID:** {results['job_run_id']}\n\n"
            f"**Training Run ID:** {results['training_run_id']}\n\n"
            f"**Pipeline Version:** {results.get('pipeline_version', 'n/a')}\n\n"
            f"**Duration:** {results.get('duration_seconds', 'N/A')}s\n\n"
            f"**New Corrections Used:** {results.get('new_corrections', 'N/A')}"
        )
        st.balloons()

        with st.expander("\U0001f4c4 Raw Results JSON", expanded=False):
            st.json(results)

        st.info("Switch to the **Model Versions** page to inspect or smoke-test the new version.")

    except Exception as e:
        progress_bar.progress(1.0, text="\u274c Retraining failed")
        status_container.update(label="\u274c Retraining failed", state="error", expanded=True)
        status_container.write(str(e))
        st.error(f"Retraining failed:\n{e}")

st.write("---")

# =====================================================================
# Step 4 \u2014 Finalize
# =====================================================================

st.header("4) \U0001f3c1 Finalize")

st.markdown("""
- The latest **trained_on** is typically the *current* version in use.  
- View, annotate, and load test models in **Model Versions**.  
- If you want to refresh model objects in this app (after retraining), use the button below.
""")

fin_col1, fin_col2 = st.columns(2)

with fin_col1:
    if st.button("\U0001f504 Refresh model caches"):
        try:
            clear_model_caches()
            st.success("Model caches cleared. Revisit Predict/Diagnostics to reload.")
        except Exception as e:
            st.error(f"Failed to clear caches: {e}")

with fin_col2:
    st.markdown("**\u26a0\ufe0f Danger zone**")
    if st.button("\U0001f5d1\ufe0f Clear All Feedback & Training Data", key="clear_data_btn"):
        st.session_state["confirm_clear"] = True

    if st.session_state.get("confirm_clear"):
        st.warning(
            "This will permanently delete **feedback.csv** and **training_data.csv**. "
            "You will lose all accumulated corrections and training data."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("\u2705 Yes, clear everything", key="confirm_clear_yes", type="primary"):
                try:
                    result = clear_feedback_and_training()
                    st.session_state["confirm_clear"] = False
                    if result["cleared"]:
                        st.success(f"Cleared: {', '.join(result['cleared'])}")
                    else:
                        st.info("Nothing to clear \u2014 files already absent.")
                except Exception as e:
                    st.error(f"Failed to clear: {e}")
        with c2:
            if st.button("\u274c Cancel", key="confirm_clear_no"):
                st.session_state["confirm_clear"] = False
