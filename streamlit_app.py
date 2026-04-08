"""Streamlit Cloud entry point — self-contained, no CWD changes.
All setup from main.py is inlined here to avoid runpy chains and
os.chdir() which break Streamlit's file watcher on multi-user sessions.
"""
import os
import sys
import logging
import types

logger = logging.getLogger("app.startup")

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.join(
    _HERE,
    "Autoclassification Scheme",
    "streamlit_demo",
    "streamlit_app",
    "src",
)

# Add app source to Python path
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# ── Step 0: Bridge Streamlit Cloud secrets → env vars ──
try:
    import streamlit as st
    for _key in [
        "DATABRICKS_HOST", "DATABRICKS_TOKEN", "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT", "TRAINING_JOB_ID",
    ]:
        if _key not in os.environ:
            try:
                os.environ[_key] = str(st.secrets[_key])
            except (KeyError, FileNotFoundError):
                pass
except Exception:
    pass

# ── Step 1: Load .env (fallback for local dev / Databricks Apps) ──
from dotenv import load_dotenv
_env_path = os.path.join(_APP_DIR, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(override=True)

# ── Step 2: Platform patches ──
host = os.environ.get("DATABRICKS_HOST", "")
if host and not host.startswith("https://"):
    os.environ["DATABRICKS_HOST"] = f"https://{host}"
os.environ.pop("DATABRICKS_CLIENT_ID", None)
os.environ.pop("DATABRICKS_CLIENT_SECRET", None)

# ── Step 3: MLflow tracking ──
import mlflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))

# ── Step 3b: NLTK data ──
try:
    import nltk
    for _pkg in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(_pkg, quiet=True)
except Exception:
    pass

# ── Step 4: sklearn compat shim ──
try:
    from sklearn.calibration import _CalibratedClassifier

    def _sklearn_compat_getattr(self, name):
        if name == "estimator" and "base_estimator" in self.__dict__:
            return self.__dict__["base_estimator"]
        if name == "base_estimator" and "estimator" in self.__dict__:
            return self.__dict__["estimator"]
        raise AttributeError(
            f"\'{type(self).__name__}\' object has no attribute \'{name}\'"
        )

    _CalibratedClassifier.__getattr__ = _sklearn_compat_getattr
except ImportError:
    pass

# ── Step 4b: Register shim modules for model unpickling ──
def _select_text(df):
    return df["text"]

_train_shim = types.ModuleType("train")
_train_shim._select_text = _select_text
sys.modules["train"] = _train_shim

try:
    from core.pipeline_utils import select_text_column  # noqa: F401
except ImportError:
    pass

# ── Step 5: Run the Streamlit app (absolute path, no CWD dependency) ──
_app_py = os.path.join(_APP_DIR, "app.py")
exec(compile(open(_app_py).read(), _app_py, "exec"))
