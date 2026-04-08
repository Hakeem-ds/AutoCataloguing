# utils/mlflow_io.py — MLflow artifact helpers (CSV format)
import os
import pandas as pd
import mlflow


def _split_remote_dir_and_name(artifact_path: str) -> tuple[str, str]:
    """Split 'artifacts/file.csv' into ('artifacts', 'file.csv')."""
    remote_dir = os.path.dirname(artifact_path).strip().strip("/\\")
    name = os.path.basename(artifact_path)
    return remote_dir, name


def log_df_artifact(df: pd.DataFrame, artifact_path: str):
    """
    Save a DataFrame as a CSV MLflow artifact.

    Example:
        log_df_artifact(df, "artifacts/step3_train_df.csv")
    """
    remote_dir, name = _split_remote_dir_and_name(artifact_path)
    if not name:
        raise ValueError(f"Invalid artifact_path (missing filename): {artifact_path}")

    local_dir = os.path.join(".mlflow_tmp_artifacts", remote_dir) if remote_dir else ".mlflow_tmp_artifacts"
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(local_dir, name)

    df.to_csv(local_file, index=False)

    if remote_dir:
        mlflow.log_artifact(local_file, artifact_path=remote_dir)
    else:
        mlflow.log_artifact(local_file)


def download_df_artifact(run_id: str, artifact_relpath: str) -> pd.DataFrame:
    """
    Download a CSV artifact from an MLflow run and return it as a DataFrame.

    Example:
        df = download_df_artifact(run_id, "artifacts/step3_train_df.csv")
    """
    mlflow.set_tracking_uri("databricks")
    uri = f"runs:/{run_id}/{artifact_relpath.lstrip('/')}"

    from mlflow.artifacts import download_artifacts
    local_path = download_artifacts(uri)

    if os.path.isdir(local_path):
        for fname in os.listdir(local_path):
            if fname.lower().endswith(".csv"):
                local_path = os.path.join(local_path, fname)
                break

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Downloaded artifact path does not exist: {local_path}")

    return pd.read_csv(local_path)
