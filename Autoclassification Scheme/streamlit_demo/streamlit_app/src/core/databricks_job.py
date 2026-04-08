# core/databricks_job.py

import time
import requests
from .config import CONFIG


class _ConfigError(RuntimeError):
    """Raised when required Databricks configuration is missing."""
    pass


def _host_token_jobid():
    """
    Validate and return Databricks host, token, and job_id.
    Ensures that required configuration exists before making API calls.
    """
    host = CONFIG.get("databricks_host")
    token = CONFIG.get("databricks_token")
    jobid = CONFIG.get("training_job_id")

    if not host:
        raise _ConfigError("Missing DATABRICKS_HOST (set in secrets or env).")
    if not token:
        raise _ConfigError("Missing DATABRICKS_TOKEN (set in secrets or env).")
    if not jobid:
        raise _ConfigError("Missing TRAINING_JOB_ID (set in secrets or env).")

    try:
        jobid = int(jobid)
    except Exception:
        raise _ConfigError(f"TRAINING_JOB_ID must be an integer; got: {jobid}")

    # Ensure scheme is present (Databricks Apps injects host without https://)
    if host and not host.startswith("http"):
        host = f"https://{host}"
    return host.rstrip("/"), token, jobid


def trigger_training_job(python_params=None):
    """
    Trigger a Databricks Job run.
    python_params should be a list of CLI args, e.g.:
        ["--queue_run_id", "...", "--queue_artifact_path", "...", "--previous_model_uri", "runs:/.../model"]
    This will be passed directly to the Databricks 'Run a Python file' task.
    """
    host, token, jobid = _host_token_jobid()

    url = f"{host}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {token}"}

    payload = {"job_id": jobid}
    if python_params:
        payload["python_params"] = python_params

    resp = requests.post(url, json=payload, headers=headers)

    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Databricks Jobs API run-now failed: {resp.text}") from e

    data = resp.json()
    run_id = data.get("run_id")

    if not run_id:
        raise RuntimeError(f"Databricks Jobs API did not return run_id: {data}")

    return run_id


def wait_for_job(run_id, poll_interval=10, on_poll=None):
    """
    Poll the Databricks job run state until completion.

    Args:
        run_id: Databricks job run ID.
        poll_interval: Seconds between polls (default 10).
        on_poll: Optional callback(life_cycle_state, elapsed_seconds)
                 called on each poll iteration. Useful for progress bars.

    Returns the full job run metadata upon SUCCESS.
    Raises RuntimeError on FAILURE or unexpected termination.
    """
    host, token, _ = _host_token_jobid()
    headers = {"Authorization": f"Bearer {token}"}

    url = f"{host}/api/2.1/jobs/runs/get"
    start_time = time.time()

    while True:
        resp = requests.get(url, params={"run_id": run_id}, headers=headers)

        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Jobs API runs/get failed: {resp.text}") from e

        data = resp.json()
        state = data.get("state", {})
        life = state.get("life_cycle_state")
        result = state.get("result_state")

        elapsed = time.time() - start_time

        # Invoke callback if provided
        if on_poll is not None:
            on_poll(life, elapsed)

        if life == "TERMINATED":
            if result == "SUCCESS":
                return data
            raise RuntimeError(f"Databricks job terminated with result={result}. Full response: {data}")

        time.sleep(poll_interval)
