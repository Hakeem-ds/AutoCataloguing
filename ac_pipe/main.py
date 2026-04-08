# main.py — Cumulative retraining pipeline
#
# Two data-source modes (supply one):
#
#   A) Feedback from MLflow (triggered by the Streamlit app):
#      python main.py --queue_run_id <id> --queue_artifact_path queued/training_data.csv
#
#   B) Local CSV bootstrap (first-ever train or ad-hoc):
#      python main.py --training_csv /path/to/training_data.csv
#
# Both modes merge cumulatively with --previous_model_uri when provided.

import argparse
import yaml
import json
import os
import hashlib
import time as _time
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from training.train import train_model
from training.evaluation import evaluate_model
from utils.mlflow_io import log_df_artifact
from utils.label_normalisation import normalise_to_taxonomy


# ================================================================
# SAFE SPLITTER — handles singleton classes correctly
# ================================================================
def safe_train_test_split_with_singletons(df, label_col, test_size, seed):
    """Split data for training/test, sending singleton classes to train only."""
    vc = df[label_col].value_counts()

    rare_labels = set(vc[vc < 2].index)
    rare_df = df[df[label_col].isin(rare_labels)].copy()
    majority_df = df[~df[label_col].isin(rare_labels)].copy()

    train_major = majority_df.copy()
    test_major = majority_df.iloc[0:0].copy()

    can_stratify = False
    if not majority_df.empty:
        uniq = majority_df[label_col].nunique()
        n = len(majority_df)
        can_stratify = (uniq >= 2) and (n >= 2 * uniq)

    if can_stratify:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for tr_idx, te_idx in sss.split(majority_df, majority_df[label_col]):
            train_major = majority_df.iloc[tr_idx].copy()
            test_major = majority_df.iloc[te_idx].copy()
    else:
        if len(majority_df) > 0:
            train_major, test_major = train_test_split(
                majority_df, test_size=test_size, random_state=seed
            )

    train_df = pd.concat([train_major, rare_df], ignore_index=True)

    # Ensure test labels are a subset of train labels
    train_labels = set(train_df[label_col].unique())
    invalid_test = ~test_major[label_col].isin(train_labels)
    if invalid_test.any():
        move_back = test_major[invalid_test].copy()
        test_major = test_major[~invalid_test].copy()
        train_df = pd.concat([train_df, move_back], ignore_index=True)

    return train_df.reset_index(drop=True), test_major.reset_index(drop=True)


# ================================================================
# Best model selection
# ================================================================
def choose_best(candidates):
    def safe_float(x, default):
        try:
            return float(x) if x is not None else default
        except Exception:
            return default

    ranked = sorted(
        candidates,
        key=lambda d: (
            -safe_float(d.get("accuracy"), float("-inf")),
            safe_float(d.get("predict_time_sec"), float("inf")),
            safe_float(d.get("fit_time_sec"), float("inf")),
        )
    )
    return ranked[0] if ranked else None


# ================================================================
# Deterministic row ID for case-insensitive deduplication
# ================================================================
def _dedup_row_id(title: str, description: str, sys_id: str) -> str:
    """
    SHA1 of lowered Title + Description + uppercased SysID.
    Matches the convention in training_data.py (_row_id) so the same
    document+label combination always produces the same hash regardless
    of casing or whitespace.
    """
    payload = f"{title.strip().lower()}||{description.strip().lower()}||{sys_id.strip().upper()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# ================================================================
# Download previous cumulative training data from MLflow
# ================================================================
def _download_previous_used_data(model_uri):
    """
    Download used_training_data.csv from the previous model's MLflow run.
    Each retrain logs this artifact so the next cycle can find it.
    Returns DataFrame[Title, Description, SysID] or None.
    """
    if not model_uri or not model_uri.startswith("runs:/"):
        return None

    try:
        from mlflow.artifacts import download_artifacts
        run_id = model_uri.split("/")[1]
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        parent_id = run.data.tags.get("mlflow.parentRunId", run_id)

        _t0 = _time.time()
        path = download_artifacts(f"runs:/{parent_id}/artifacts/used_training_data.csv")
        _dl_sec = _time.time() - _t0

        df = pd.read_csv(path)
        if not df.empty and "SysID" in df.columns:
            print(f"  \U0001F4E6 Previous training data: {len(df)} rows (downloaded in {_dl_sec:.1f}s)")
            return df[["Title", "Description", "SysID"]]
        return None
    except Exception as e:
        print(f"  \u26a0 Could not download previous training data: {e}")
        return None


# ================================================================
# Load new data from either source
# ================================================================
def _load_new_data(args):
    """
    Load new training data from one of two sources:
      A) MLflow feedback artifact (--queue_run_id + --queue_artifact_path)
      B) Local CSV file         (--training_csv)
    Returns DataFrame with at least [Title, Description, SysID].
    """
    if args.queue_run_id and args.queue_artifact_path:
        from mlflow.artifacts import download_artifacts
        print(f"\U0001F4E5 Downloading feedback CSV: {args.queue_run_id}/{args.queue_artifact_path}")
        _t0 = _time.time()
        csv_path = download_artifacts(
            f"runs:/{args.queue_run_id}/{args.queue_artifact_path}"
        )
        df = pd.read_csv(csv_path)
        print(f"  \u2192 {len(df)} rows loaded in {_time.time() - _t0:.1f}s")
        mlflow.set_tag("data_source", "mlflow_feedback")
        mlflow.set_tag("queue_run_id", args.queue_run_id)

    elif args.training_csv:
        print(f"\U0001F4E5 Loading training CSV: {args.training_csv}")
        _t0 = _time.time()
        df = pd.read_csv(args.training_csv)
        print(f"  \u2192 {len(df)} rows loaded in {_time.time() - _t0:.1f}s")
        mlflow.set_tag("data_source", "local_csv")
        mlflow.set_tag("training_csv", args.training_csv)

    else:
        raise RuntimeError(
            "No data source. Provide either:\n"
            "  --queue_run_id + --queue_artifact_path  (feedback from MLflow)\n"
            "  --training_csv /path/to/data.csv         (bootstrap / ad-hoc)"
        )

    if df.empty:
        raise ValueError("Training data is empty.")
    return df


# ================================================================
# Build cumulative train/test
# ================================================================
def build_train_test(args, cfg):
    """
    Cumulative retraining: loads new data, optionally merges with previous
    model's training data, deduplicates, and splits.
    """
    _phase_start = _time.time()

    # ── Phase 1: Load data ──
    _t0 = _time.time()
    new_data = _load_new_data(args)
    _load_sec = _time.time() - _t0

    # ── Phase 2: Cumulative merge ──
    _t0 = _time.time()
    prev_data = _download_previous_used_data(args.previous_model_uri)

    if prev_data is not None and not prev_data.empty:
        combined = pd.concat(
            [prev_data, new_data[["Title", "Description", "SysID"]]],
            ignore_index=True
        )
        combined = combined.drop_duplicates(
            subset=["Title", "Description", "SysID"]
        ).reset_index(drop=True)
        print(f"  \U0001F4CA Cumulative: {len(prev_data)} previous + {len(new_data)} new \u2192 {len(combined)} unique rows")
    else:
        combined = new_data
        print(f"  \U0001F4CA Training on {len(combined)} rows (no previous data to merge)")
    _merge_sec = _time.time() - _t0

    combined["Title"] = combined["Title"].fillna("")
    combined["Description"] = combined["Description"].fillna("")
    combined["combined_text"] = combined["Title"] + " " + combined["Description"]

    # ── Phase 3: Normalise labels to taxonomy ──
    # Corrects zero-padding (LT0258->LT000258), end-zero typos (LT00032->LT000320),
    # and file-level indexes (/002/007->/002). Uses folder_name_map as the taxonomy.
    # Loads from co-located artifacts/ dir — no dependency on Streamlit app tree.
    _t0 = _time.time()
    try:
        _fmap_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "artifacts", "folder_name_map.json"
        )
        if os.path.exists(_fmap_path):
            with open(_fmap_path) as _f:
                _fmap = json.load(_f)
            _taxonomy = set(_fmap.keys())
            before_classes = combined["SysID"].nunique()
            # OPT: resolve unique SysIDs only, then map back
            _raw_sysids = combined["SysID"].astype(str).str.strip()
            _unique_sysids = _raw_sysids.unique()
            _label_cache = {s: normalise_to_taxonomy(s, _taxonomy) for s in _unique_sysids}
            combined["SysID"] = _raw_sysids.map(_label_cache)
            after_classes = combined["SysID"].nunique()
            _norm_sec = _time.time() - _t0
            print(f"  \U0001F3F7\uFE0F Label normalisation: {before_classes} \u2192 {after_classes} classes ({len(_unique_sysids)} unique resolved in {_norm_sec:.1f}s)")
            mlflow.set_tag("label_normalisation", "taxonomy_v3.0")
            mlflow.log_metric("classes_before_norm", before_classes)
            mlflow.log_metric("classes_after_norm", after_classes)
        else:
            print(f"  \u26a0 folder_name_map.json not found at {_fmap_path} \u2014 skipping label normalisation")
    except Exception as _e:
        print(f"  \u26a0 Label normalisation skipped: {_e}")
    _norm_sec = _time.time() - _t0

    # ── Phase 4: Post-normalisation case-insensitive dedup ──
    # The earlier raw dedup is case-sensitive and runs BEFORE label normalisation.
    # This final pass catches:
    #   (a) different SysID formats that normalise to the same label
    #   (b) case variations in Title/Description
    #   (c) whitespace differences (leading/trailing)
    # OPT: vectorized string concat instead of row-by-row .apply() with SHA1.
    # The dedup key is temporary (dropped after dedup), so plain concat is
    # sufficient and ~10-50x faster than per-row SHA1 hashing.
    _t0 = _time.time()
    combined["_dedup_key"] = (
        combined["Title"].astype(str).str.strip().str.lower() + "||" +
        combined["Description"].astype(str).str.strip().str.lower() + "||" +
        combined["SysID"].astype(str).str.strip().str.upper()
    )
    before_final_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"]).reset_index(drop=True)
    after_final_dedup = len(combined)
    _dedup_sec = _time.time() - _t0
    if before_final_dedup > after_final_dedup:
        removed = before_final_dedup - after_final_dedup
        print(f"  \U0001F9F9 Post-normalisation dedup: {before_final_dedup} \u2192 {after_final_dedup} rows ({removed} removed, {_dedup_sec:.2f}s)")
        mlflow.log_metric("post_norm_duplicates_removed", removed)
    else:
        print(f"  \u2705 Post-normalisation dedup: no duplicates ({after_final_dedup} rows, {_dedup_sec:.2f}s)")

    combined["New SysID"] = combined["SysID"].astype(str)

    # ── Phase 5: Split ──
    _t0 = _time.time()
    test_size = cfg["split"]["test_size"]
    seed = cfg["split"]["seed"]
    train_df, test_df = safe_train_test_split_with_singletons(
        combined, "New SysID", test_size, seed
    )
    _split_sec = _time.time() - _t0

    log_df_artifact(train_df, "artifacts/step3_train_df.csv")
    log_df_artifact(test_df, "artifacts/step3_test_df.csv")

    # ── Log phase timings ──
    _total_sec = _time.time() - _phase_start
    print(f"\n  \u23f1 build_train_test timings:")
    print(f"    load={_load_sec:.1f}s  merge={_merge_sec:.1f}s  norm={_norm_sec:.1f}s  dedup={_dedup_sec:.2f}s  split={_split_sec:.1f}s  TOTAL={_total_sec:.1f}s")
    mlflow.log_metric("build_data_seconds", round(_total_sec, 1))
    mlflow.log_metric("phase_load_seconds", round(_load_sec, 1))
    mlflow.log_metric("phase_merge_seconds", round(_merge_sec, 1))
    mlflow.log_metric("phase_norm_seconds", round(_norm_sec, 1))
    mlflow.log_metric("phase_dedup_seconds", round(_dedup_sec, 2))
    mlflow.log_metric("phase_split_seconds", round(_split_sec, 1))

    return train_df, test_df



# ================================================================
# Fetch previous model metrics for improvement tracking
# ================================================================
def _log_previous_metrics(model_uri):
    """
    Fetch the previous model's key metrics from MLflow and log them as tags
    on the current run, so the model_versions page can show improvement deltas.
    """
    if not model_uri or not model_uri.startswith("runs:/"):
        return {}
    try:
        run_id = model_uri.split("/")[1]
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        parent_id = run.data.tags.get("mlflow.parentRunId")

        # The candidate run has the metrics; if this IS the candidate, use it directly
        metrics = dict(run.data.metrics or {})
        if not metrics and parent_id:
            # Try the parent (pipeline run) — but metrics are on the candidate
            pass

        prev = {}
        for key in ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
                     "weighted_precision", "mean_confidence", "n_training_samples"]:
            if key in metrics:
                prev[key] = float(metrics[key])
                mlflow.set_tag(f"prev_{key}", f"{metrics[key]:.6f}")

        if prev:
            print(f"  \U0001F4CA Previous model metrics logged: {list(prev.keys())}")
        return prev
    except Exception as e:
        print(f"  \u26a0 Could not fetch previous metrics: {e}")
        return {}

# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Cumulative retraining pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")

    # Data source A: MLflow feedback artifacts
    parser.add_argument("--queue_run_id", type=str, default=None,
                        help="MLflow run_id containing the queued training CSV")
    parser.add_argument("--queue_artifact_path", type=str, default=None,
                        help="Artifact path, e.g. 'queued/training_data.csv'")

    # Data source B: local CSV (bootstrap / ad-hoc)
    parser.add_argument("--training_csv", type=str, default=None,
                        help="Path to a CSV with columns: Title, Description, SysID")

    # Previous model for cumulative data merge
    parser.add_argument("--previous_model_uri", type=str, default=None,
                        help="MLflow URI of the previous model for cumulative data merge")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name="full_pipeline"):
        mlflow.set_tag("pipeline_version", cfg["pipeline_version"])

        # ── Build cumulative train/test ──
        train_df, test_df = build_train_test(args, cfg)

        # ── Log previous model metrics for comparison ──
        _log_previous_metrics(args.previous_model_uri)

        # ── Log cumulative data for next retrain cycle ──
        _all_data = pd.concat([train_df, test_df], ignore_index=True)
        _used = pd.DataFrame({
            "Title": _all_data["Title"].fillna("") if "Title" in _all_data.columns else "",
            "Description": _all_data["Description"].fillna("") if "Description" in _all_data.columns else "",
            "SysID": _all_data["New SysID"].astype(str) if "New SysID" in _all_data.columns
                     else _all_data["SysID"].astype(str),
        })
        log_df_artifact(_used, "artifacts/used_training_data.csv")
        print(f"\U0001F4E6 Logged used_training_data.csv ({len(_used)} rows) for next retrain cycle")

        # ── Prepare model inputs ──
        train_X = pd.DataFrame({"text": train_df["combined_text"].astype(str)})
        y_train = train_df["New SysID"].astype(str)
        test_X = pd.DataFrame({"text": test_df["combined_text"].astype(str)})
        y_test = test_df["New SysID"].astype(str)

        results = []

        # ── Train candidates ──
        for m in cfg["models"]:
            name = m["name"]
            print(f"\n=== Training candidate: {name} ===")

            with mlflow.start_run(nested=True, run_name=f"candidate::{name}"):

                pipeline, fit_time, meta = train_model(
                    model_type=m["type"],
                    train_df=train_X,
                    y_train=y_train,
                    tfidf_params=m["params"]["tfidf"],
                    calibration=m["params"]["calibration"],
                    svm=m["params"].get("svm"),
                )

                if meta and isinstance(meta, dict):
                    if "model_params" in meta:
                        mlflow.log_params(meta["model_params"])
                    if "fit_time_sec" in meta:
                        mlflow.log_metric("fit_time_sec", meta["fit_time_sec"])

                # Evaluate
                if len(test_X) > 0:
                    eval_metrics = evaluate_model(
                        pipeline, test_X, y_test,
                        train_df=train_X, y_train=y_train,
                    )
                    loggable = {
                        k: float(v) for k, v in eval_metrics.items()
                        if isinstance(v, (int, float)) and v is not None
                    }
                    mlflow.log_metrics(loggable)
                else:
                    eval_metrics = {}
                    mlflow.set_tag("evaluation_skipped", "empty_test_set")
                    print("  \u26a0 Test set is empty \u2014 evaluation skipped")

                # Note: model already logged by train_model() with signature

                result = {
                    "name": name,
                    "accuracy": eval_metrics.get("accuracy"),
                    "fit_time_sec": meta.get("fit_time_sec") if meta else None,
                    "predict_time_sec": eval_metrics.get("predict_time_sec"),
                    "run_id": mlflow.active_run().info.run_id,
                }
                results.append(result)

        # ── Select best model ──
        best = choose_best(results)
        if best:
            mlflow.set_tag("best_model", best["name"])
            mlflow.set_tag("best_model_run_id", best["run_id"])
            best_name = best["name"]
            best_rid = best["run_id"]
            print(f"\n\U0001F3C6 Best model: {best_name} (run_id: {best_rid})")
        else:
            print("\nNo valid model found.")

        # ── Save results ──
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        log_df_artifact(pd.DataFrame(results), "artifacts/results.csv")

if __name__ == "__main__":
    main()
