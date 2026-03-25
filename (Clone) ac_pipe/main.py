# main.py
import argparse
import yaml
import json
import time
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import load_multiple_excel_files
from transform import transform_excel, process_colored_dfs
from preprocessing import clean_text_column
from train import train_model
from evaluation import evaluate_model

from mlflow_io import (
    log_df_artifact,
    download_df_artifact,
    find_latest_successful_run,
)

# ==============================================
# Best Model Selection Helper
# ==============================================
def choose_best(candidates):
    """
    Rank by highest accuracy, then lowest predict_time, then lowest fit_time.
    """
    ranked = sorted(
        candidates,
        key=lambda d: (
            -d.get("accuracy", 0.0),
            d.get("predict_time_sec", float("inf")),
            d.get("fit_time_sec", float("inf"))
        )
    )
    return ranked[0] if ranked else None


# ==============================================
# STEP 1: Load Excel Datasets
# ==============================================
def step1_load(cfg, resume_ctx):
    if resume_ctx.get("resume_run_id"):
        try:
            full_df = download_df_artifact(
                resume_ctx["resume_run_id"], "artifacts/step1_full_df.parquet"
            )
            folder_mapping = download_df_artifact(
                resume_ctx["resume_run_id"], "artifacts/step1_folder_mapping.parquet"
            )
            print("Step 1: Resumed from artifacts.")
            return full_df, folder_mapping
        except Exception:
            print("Step 1: Resume failed; recomputing.")

    print("\n📥 STEP 1: Loading datasets...")

    paths = cfg["data"]
    file_configs = [
        {"path": paths["wholly_amalgamated"], "mode": "standard", "name": "wholly_amalgamated"},
        {"path": paths["split_series"], "mode": "colored", "name": "split_series"}
    ]

    loaded = load_multiple_excel_files(file_configs)
    df_standard = loaded["wholly_amalgamated"]["df_standard"]
    dfs_colored = loaded["split_series"]

    folder_mapping = transform_excel(df_standard)     # reference/name mapping
    full_df = process_colored_dfs(dfs_colored)        # combined labeled rows

    log_df_artifact(full_df, "artifacts/step1_full_df.parquet")
    log_df_artifact(folder_mapping, "artifacts/step1_folder_mapping.parquet")

    return full_df, folder_mapping


# ==============================================
# STEP 2: Clean + Filter
# ==============================================
def step2_clean_and_filter(full_df, folder_mapping, resume_ctx):
    if resume_ctx.get("resume_run_id"):
        try:
            filtered = download_df_artifact(
                resume_ctx["resume_run_id"], "artifacts/step2_filtered.parquet"
            )
            print("Step 2: Resumed from artifacts.")
            return filtered
        except Exception:
            print("Step 2: Resume failed; recomputing.")

    print("\n🧼 STEP 2: Cleaning & filtering...")

    # Example filter: keep IDs containing "LT0"
    filtered_df = full_df[full_df["New SysID"].str.contains("LT0", na=False)]

    folder_name_map = dict(zip(folder_mapping["Reference"], folder_mapping["Name"]))
    filtered_df["folder_name"] = filtered_df["New SysID"].map(folder_name_map)

    # Require at least 2 samples per class
    vc = filtered_df["New SysID"].value_counts()
    ids_with_2plus = vc[vc >= 2].index
    filtered_df = filtered_df[filtered_df["New SysID"].isin(ids_with_2plus)].copy()

    # Clean & combine text
    filtered_df["clean_title"] = clean_text_column(filtered_df, "Title")
    filtered_df["clean_description"] = clean_text_column(filtered_df, "Description")
    filtered_df["combined_text"] = filtered_df["clean_title"] + " " + filtered_df["clean_description"]

    log_df_artifact(filtered_df, "artifacts/step2_filtered.parquet")
    return filtered_df


# ==============================================
# STEP 3: Train/Test Split
# ==============================================
def step3_split(filtered_df, cfg, resume_ctx):
    if resume_ctx.get("resume_run_id"):
        try:
            train_df = download_df_artifact(
                resume_ctx["resume_run_id"], "artifacts/step3_train_df.parquet"
            )
            test_df = download_df_artifact(
                resume_ctx["resume_run_id"], "artifacts/step3_test_df.parquet"
            )
            print("Step 3: Resumed from artifacts.")
            return train_df, test_df
        except Exception:
            print("Step 3: Resume failed; recomputing.")

    print("\n✂️ STEP 3: Creating stratified split...")

    X = filtered_df
    test_size = cfg["split"]["test_size"]
    seed = cfg["split"]["seed"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for tr_idx, te_idx in sss.split(X, X["New SysID"]):
        train_df = X.iloc[tr_idx].copy()
        test_df = X.iloc[te_idx].copy()

    # Ensure test classes subset coverage
    test_classes = set(test_df["New SysID"])
    train_df = train_df[train_df["New SysID"].isin(test_classes)].copy()

    log_df_artifact(train_df, "artifacts/step3_train_df.parquet")
    log_df_artifact(test_df, "artifacts/step3_test_df.parquet")
    return train_df, test_df


# ==============================================
# Feedback-based data loader
# ==============================================
def build_train_test_from_feedback_artifact(queue_run_id: str, artifact_relpath: str, cfg):
    """
    Download a queued training CSV logged as an MLflow artifact at:
      runs:/<queue_run_id>/<artifact_relpath>
    Expect columns: Title, Description, SysID
    Build train_df/test_df with combined_text and New SysID for downstream pipeline.
    """
    if not queue_run_id or not artifact_relpath:
        raise RuntimeError("--use_feedback requires both --queue_run_id and --queue_artifact_path")

    from mlflow.artifacts import download_artifacts
    print(f"📥 Using queued feedback artifact: run={queue_run_id}, path={artifact_relpath}")
    local_csv = download_artifacts(f"runs:/{queue_run_id}/{artifact_relpath}")
    tdf = pd.read_csv(local_csv)
    if tdf.empty:
        raise ValueError("Queued training_data.csv is empty.")

    # Normalize schema for pipeline usage
    tdf["Title"] = tdf["Title"].fillna("")
    tdf["Description"] = tdf["Description"].fillna("")
    tdf["combined_text"] = tdf["Title"] + " " + tdf["Description"]
    tdf["New SysID"] = tdf["SysID"].astype(str)

    # Stratified split
    test_size = cfg["split"]["test_size"]
    seed = cfg["split"]["seed"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for tr_idx, te_idx in sss.split(tdf, tdf["New SysID"]):
        train_df = tdf.iloc[tr_idx].copy()
        test_df = tdf.iloc[te_idx].copy()

    # Optional: log the exact CSV used for reproducibility
    try:
        import shutil
        shutil.copy(local_csv, "used_training_data.csv")
        mlflow.log_artifact("used_training_data.csv")
        mlflow.set_tag("data_source", "queued_feedback_artifact")
        mlflow.set_tag("queue_run_id", queue_run_id)
    except Exception:
        pass

    # Also persist train/test for later resume if desired
    log_df_artifact(train_df, "artifacts/step3_train_df.parquet")
    log_df_artifact(test_df, "artifacts/step3_test_df.parquet")

    return train_df, test_df


# ==============================================
# MAIN PIPELINE
# ==============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from_step", default="all", choices=["all", "train"])
    parser.add_argument("--config", type=str, default="config.yaml")

    # Feedback-based retraining flags
    parser.add_argument("--use_feedback", action="store_true",
                        help="Use queued training CSV from MLflow artifacts instead of Excel sources.")
    parser.add_argument("--queue_run_id", type=str, default=None,
                        help="MLflow run_id that contains queued/training_data.csv")
    parser.add_argument("--queue_artifact_path", type=str, default=None,
                        help="Artifact relative path for training_data.csv, e.g. 'queued/training_data.csv'")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Ensure we are using Databricks tracking (in Jobs & notebooks)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name="full_pipeline") as parent_run:

        mlflow.set_tag("pipeline_version", cfg["pipeline_version"])

        resume_ctx = {"resume_run_id": None}

        # Resume logic: find a matching previous successful run
        if args.resume:
            latest = find_latest_successful_run(
                cfg["experiment"], "pipeline_version", cfg["pipeline_version"]
            )
            if latest:
                resume_ctx["resume_run_id"] = latest.info.run_id
                print(f"Resuming from run {resume_ctx['resume_run_id']}")

        # Build train/test depending on mode
        if args.use_feedback:
            # Always rebuild from queued artifact when requested
            train_df, test_df = build_train_test_from_feedback_artifact(
                queue_run_id=args.queue_run_id,
                artifact_relpath=args.queue_artifact_path,
                cfg=cfg,
            )

        else:
            # Steps 1–3 path (Excel → clean → split), with proper resume/fresh handling
            if args.from_step == "all":
                full_df, folder_mapping = step1_load(cfg, resume_ctx)
                filtered_df = step2_clean_and_filter(full_df, folder_mapping, resume_ctx)
                train_df, test_df = step3_split(filtered_df, cfg, resume_ctx)

            else:
                # Load train/test via resume if available; otherwise rebuild fresh
                resume_run = resume_ctx.get("resume_run_id")

                if resume_run:
                    print(f"🔄 Resuming from run_id={resume_run}")
                    train_df = download_df_artifact(resume_run, "artifacts/step3_train_df.parquet")
                    test_df  = download_df_artifact(resume_run, "artifacts/step3_test_df.parquet")
                else:
                    print("🆕 Fresh training run — no resume context found")
                    full_df, folder_mapping = step1_load(cfg, resume_ctx)
                    filtered_df = step2_clean_and_filter(full_df, folder_mapping, resume_ctx)
                    train_df, test_df = step3_split(filtered_df, cfg, resume_ctx)

        # Prepare model inputs
        train_X = pd.DataFrame({"text": train_df["combined_text"].astype(str)})
        y_train = train_df["New SysID"].astype(str)
        test_X  = pd.DataFrame({"text": test_df["combined_text"].astype(str)})
        y_test  = test_df["New SysID"].astype(str)

        model_candidates = cfg["models"]
        results = []

        # Loop over candidate models
        for m in model_candidates:
            name = m["name"]
            print(f"\n=== Training candidate: {name} ===")

            with mlflow.start_run(nested=True, run_name=f"candidate::{name}") as child:

                pipeline, fit_time, meta = train_model(
                    model_type=m["type"],
                    train_df=train_X,
                    y_train=y_train,
                    tfidf_params=m["params"]["tfidf"],
                    calibration=m["params"]["calibration"],
                    svm=m["params"].get("svm"),
                )

                out = evaluate_model(pipeline, test_X, y_test)

                run_id = mlflow.active_run().info.run_id
                # predict_time_sec may not be set by evaluate_model depending on your impl
                predict_time_metric = mlflow.get_run(run_id).data.metrics.get("predict_time_sec", float("inf"))

                summary = {
                    "name": name,
                    "run_id": run_id,
                    "model_uri": f"runs:/{run_id}/model",
                    "fit_time_sec": float(fit_time) if fit_time is not None else float("inf"),
                    "predict_time_sec": float(predict_time_metric),
                    "accuracy": float(out["accuracy"]),
                    "macro_f1": float(out["report"]["macro avg"]["f1-score"]),
                    "weighted_f1": float(out["report"]["weighted avg"]["f1-score"]),
                }
                results.append(summary)

        # Select best model and log manifest at parent level
        best = choose_best(results)

        if best:
            manifest = {
                "timestamp": time.time(),
                "pipeline_version": cfg["pipeline_version"],
                "best": best,
                "candidates": results,
            }
            with open("model_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            mlflow.log_artifact("model_manifest.json")

        print("\n=== Model Comparison ===")
        for r in results:
            print(r)

        if best:
            print(f"\n🏆 Best Model: {best['name']}")


if __name__ == "__main__":
    main()