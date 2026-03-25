# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install "pyarrow>=10.0.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 1
#New run
import sys
from main import main

# Force clean CLI args for Databricks
sys.argv = ["main.py", "--config", "config.yaml"]

main()

# COMMAND ----------

!pip freeze > training_requirements.txt

# COMMAND ----------

#Resumed run

# import sys
# from main import main

# sys.argv = ["main.py", "--config", "config.yaml", "--resume", "--from_step", "train"]

# main()

# COMMAND ----------

# DBTITLE 1,Cell 3
# MAGIC %pip install "typer<0.10.0"
# MAGIC %pip install -U spacy
# MAGIC %pip install -U spacy && python -m spacy download en_core_web_sm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

def walk(run_id, path="", indent=0):
    items = client.list_artifacts(run_id, path)
    for item in items:
        print(" " * indent, item.path)
        if item.is_dir:
            walk(run_id, item.path, indent+2)

walk("be7948775d474298b19aa67fbc756ab9")

# COMMAND ----------


