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
st.cache_data.clear()
st.cache_resource.clear()

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


