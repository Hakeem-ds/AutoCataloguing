# mlflow_io.py
import os
import tempfile
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

def log_df_artifact(df: pd.DataFrame, artifact_relpath: str):
    """Log a DataFrame as a Parquet artifact under the current active run."""
    tmpdir = tempfile.mkdtemp()
    local_path = os.path.join(tmpdir, artifact_relpath)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_parquet(local_path, index=False)
    mlflow.log_artifact(local_path, artifact_path=os.path.dirname(artifact_relpath))

def log_json_artifact(obj, artifact_relpath: str):
    import json
    tmpdir = tempfile.mkdtemp()
    local_path = os.path.join(tmpdir, artifact_relpath)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(obj, f, indent=2)
    mlflow.log_artifact(local_path, artifact_path=os.path.dirname(artifact_relpath))

def download_df_artifact(run_id: str, artifact_relpath: str) -> pd.DataFrame:
    """Download a Parquet artifact from a run and return as DataFrame."""
    # Example URI: runs:/<run_id>/<artifact_relpath>
    uri = f"runs:/{run_id}/{artifact_relpath}"
    local_dir = mlflow.artifacts.download_artifacts(uri)
    # If artifact_relpath refers to a file, local_dir may be the file path itself.
    # Ensure we read the correct file:
    if os.path.isfile(local_dir):
        return pd.read_parquet(local_dir)
    # Otherwise, try to find the file within the directory
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".parquet"):
                return pd.read_parquet(os.path.join(root, f))
    raise FileNotFoundError(f"Could not find Parquet artifact at {uri}")

def find_latest_successful_run(experiment_name: str, tag_key: str, tag_value: str):
    """Find the latest run in an experiment that has tag_key=tag_value and status FINISHED."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and tags.{tag_key} = '{tag_value}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    return runs[0] if runs else None