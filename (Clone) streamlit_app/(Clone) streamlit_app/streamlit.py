# Databricks notebook source
!pip install pyngrok streamlit python-dotenv

# COMMAND ----------

import signal
import os
import time
import subprocess
import dotenv
from pyngrok import ngrok

# COMMAND ----------

# Set ngrok auth token
def configure_ngrok():
    dotenv.load_dotenv()
    os.environ['NGROK_AUTH_TOKEN'] = os.getenv('NGROK_AUTH_TOKEN')
    ngrok.set_auth_token(os.environ['NGROK_AUTH_TOKEN'])

configure_ngrok()


# COMMAND ----------

# Launch Streamlit
process = subprocess.Popen([
    "python", "-m", "streamlit", "run", "app.py", "--server.port", "8501"
])

# COMMAND ----------

# Wait for Streamlit to boot
time.sleep(5)

# Open tunnel
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url.public_url}")

# COMMAND ----------

import streamlit as st
st.cache_resource.clear()

# COMMAND ----------

import mlflow

run_id = "9ff12cd3edd94364a5e802811b9f3bfa"   # your actual run_id
run = mlflow.get_run(run_id)

print(run.info.artifact_uri)

# COMMAND ----------

import mlflow
import pandas as pd

mlflow_model_uri = "runs:/9ff12cd3edd94364a5e802811b9f3bfa/model"

pipe = mlflow.sklearn.load_model(mlflow_model_uri)

# IMPORTANT: The pipeline expects a DataFrame with a 'text' column
df = pd.DataFrame({"text": ["test document about lift outage and escalator maintenance"]})
print(pipe.predict(df))

# COMMAND ----------

# client.search_runs(experiment_ids=runs:/9ff12cd3edd94364a5e802811b9f3bfa/model, filter_string='', max_results=10)
client.search_runs(experiment_ids=["1991163990441594"], filter_string='', max_results=10)

# COMMAND ----------

# Update artifacts/model_registry.json to point "unversioned" at your MLflow model
from core.model_registry import get_model_registry, save_model_registry
from core.config import CONFIG

mlflow_model_uri = "runs:/9ff12cd3edd94364a5e802811b9f3bfa/model"

reg = get_model_registry()
reg["unversioned"] = {
    "mlflow_model_uri": mlflow_model_uri,
    "trained_on": "2026-02-25T00:00:00Z",  # you can set now() or a real timestamp
    "notes": "First MLflow sklearn pipeline model via DBFS URI",
    # Optional: keep this so taxonomy can still be loaded
    "folder_name_map": CONFIG.get("folder_name_map")
}
save_model_registry(reg)

print("Registry updated. Current keys:", list(reg.keys()))
print("Unversioned entry:", reg["unversioned"])

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

parent_run_id = "be7948775d474298b19aa67fbc756ab9"   # your parent full_pipeline run
exp_id = "1991163990441594"                          # your experiment ID

runs = client.search_runs(
    experiment_ids=[exp_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
)

for r in runs:
    print("CHILD RUN:", r.info.run_id, r.data.tags.get("mlflow.runName"))

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

def walk(run_id, path="", indent=0):
    items = client.list_artifacts(run_id, path)
    for item in items:
        print(" " * indent, item.path)
        if item.is_dir:
            walk(run_id, item.path, indent+2)

walk("9ff12cd3edd94364a5e802811b9f3bfa")

# COMMAND ----------

# Kill streamlit process
for proc in subprocess.check_output(['ps', 'aux']).decode().split('\n'):
    if 'streamlit' in proc and 'python' in proc:
        pid = int(proc.split()[1])
        os.kill(pid, signal.SIGTERM)

# COMMAND ----------

# Kill ngrok process
for proc in subprocess.check_output(['ps', 'aux']).decode().split('\n'):
    if 'ngrok' in proc and 'python' not in proc:
        pid = int(proc.split()[1])
        os.kill(pid, signal.SIGTERM)



# COMMAND ----------

import time

# Ping the cluster every 30 minutes to keep it alive (less than 44 minutes)
while True:
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    time.sleep(30 * 60)

# COMMAND ----------

print('h')

# COMMAND ----------


