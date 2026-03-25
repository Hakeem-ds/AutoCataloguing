# core/databricks_job.py
import time
import requests
from .config import CONFIG

class _ConfigError(RuntimeError): pass

def _host_token_jobid():
    host  = CONFIG.get("databricks_host")
    token = CONFIG.get("databricks_token")
    jobid = CONFIG.get("training_job_id")
    if not host:  raise _ConfigError("Missing DATABRICKS_HOST")
    if not token: raise _ConfigError("Missing DATABRICKS_TOKEN")
    if not jobid: raise _ConfigError("Missing TRAINING_JOB_ID")
    try:
        jobid = int(jobid)
    except Exception:
        raise _ConfigError(f"TRAINING_JOB_ID must be an integer; got: {jobid}")
    return host.rstrip("/"), token, jobid

def trigger_training_job(python_params: list[str] | None = None):
    host, token, jobid = _host_token_jobid()
    url = f"{host}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"job_id": jobid}

    # Only valid if your Job task type is "Python file"
    if python_params:
        payload["python_params"] = python_params

    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Jobs API run-now failed: {resp.text}") from e

    return resp.json()["run_id"]

def wait_for_job(run_id, poll_interval=10):
    host, token, _ = _host_token_jobid()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{host}/api/2.1/jobs/runs/get"
    while True:
        resp = requests.get(url, params={"run_id": run_id}, headers=headers)
        resp.raise_for_status()
        state = resp.json().get("state", {})
        life = state.get("life_cycle_state")
        result = state.get("result_state")
        if life == "TERMINATED":
            if result == "SUCCESS":
                return resp.json()
            raise RuntimeError(f"Databricks job terminated with result={result}. Full: {resp.text}")
        time.sleep(poll_interval)