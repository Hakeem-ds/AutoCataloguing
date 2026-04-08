"""App entry point — works on Databricks Apps AND Streamlit Community Cloud.
Loads secrets/env, patches platform vars, applies sklearn compat shims,
validates deps, downloads NLTK data if needed, runs app.
"""
import os
import sys
import logging

logger = logging.getLogger("app.startup")

# ── Step 0: Bridge Streamlit Cloud secrets → env vars ──
# On Streamlit Community Cloud, secrets live in st.secrets (secrets.toml).
# On Databricks Apps, secrets come via .env / app secrets (os.environ).
# This bridge ensures the rest of the app can always use os.environ.
try:
    import streamlit as st
    _SECRETS_KEYS = [
        "DATABRICKS_HOST", "DATABRICKS_TOKEN", "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT", "TRAINING_JOB_ID",
    ]
    for _key in _SECRETS_KEYS:
        if _key not in os.environ:
            try:
                os.environ[_key] = str(st.secrets[_key])
            except (KeyError, FileNotFoundError):
                pass  # Secret not set on this platform — dotenv will try next
    logger.info("Streamlit Cloud secrets bridge applied")
except Exception:
    pass  # st.secrets unavailable (e.g. during import-only or tests)

# ── Step 1: Load .env (explicit path, override=True so .env wins over platform vars) ──
# Use the script's own directory to find .env, not CWD (which differs
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)
    logger.info(f"Loaded .env from {_env_path}")
else:
    load_dotenv(override=True)  # fallback: search CWD
    logger.warning(f".env not found at {_env_path}, falling back to CWD")

# ── Step 2: Platform compatibility patches ──
host = os.environ.get("DATABRICKS_HOST", "")
if host and not host.startswith("https://"):
    os.environ["DATABRICKS_HOST"] = f"https://{host}"

os.environ.pop("DATABRICKS_CLIENT_ID", None)
os.environ.pop("DATABRICKS_CLIENT_SECRET", None)

# ── Step 3: Centralised MLflow tracking URI ──
import mlflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))

# ── Step 3b: Download NLTK data if not present ──
# On Databricks Apps / Docker, NLTK data is pre-installed at build time.
# On Streamlit Cloud, there's no Dockerfile, so download on first run.
try:
    import nltk
    # Download directly — idempotent and avoids BadZipFile on corrupt caches
    for _pkg in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(_pkg, quiet=True)
except Exception:
    pass  # NLTK not critical for app startup

# ── Step 4: sklearn backward-compat shim ──
# Models pickled on sklearn 1.1.1 have _CalibratedClassifier objects with
# 'base_estimator'. sklearn 1.2.x renamed it to 'estimator' but the internal
# class _CalibratedClassifier doesn't always bridge both names on unpickled
# objects. This shim transparently maps between the two.
try:
    from sklearn.calibration import _CalibratedClassifier

    def _sklearn_compat_getattr(self, name):
        if name == "estimator" and "base_estimator" in self.__dict__:
            return self.__dict__["base_estimator"]
        if name == "base_estimator" and "estimator" in self.__dict__:
            return self.__dict__["estimator"]
        raise AttributeError(f"\'{type(self).__name__}\' object has no attribute \'{name}\'")

    _CalibratedClassifier.__getattr__ = _sklearn_compat_getattr
    logger.info("sklearn compat shim applied: _CalibratedClassifier base_estimator <-> estimator")
except ImportError:
    pass

# ── Step 4b: Register shim modules for model unpickling ──
# Models trained by ac_pipe/train.py pickle FunctionTransformer references
# to train._select_text. The app runtime has no 'train' module, so we
# register a shim module that provides the function.
import types
import runpy

def _select_text(df):
    """Extract 'text' column from DataFrame — pickle-safe text selector."""
    return df["text"]

_train_shim = types.ModuleType("train")
_train_shim._select_text = _select_text
sys.modules["train"] = _train_shim
logger.info("Registered 'train' shim module for model unpickling")

# Also ensure core.pipeline_utils.select_text_column is available
# (used by re-logged models)
try:
    from core.pipeline_utils import select_text_column  # noqa: F401
    logger.info("core.pipeline_utils.select_text_column loaded")
except ImportError:
    pass

# ── Step 5: Validate model dependencies at startup ──
def validate_model_deps():
    try:
        from importlib.metadata import version as pkg_version
        import yaml, json

        registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "model_registry.json")
        with open(registry_path) as f:
            registry = json.load(f)

        meta = registry.get("__meta__", {})
        promoted_key = meta.get("promoted")
        if not promoted_key or promoted_key not in registry:
            return

        model_uri = registry[promoted_key].get("mlflow_model_uri", "")
        if not model_uri.startswith("runs:/"):
            return

        run_id = model_uri.split("/")[1]
        client = mlflow.tracking.MlflowClient()
        conda_path = client.download_artifacts(run_id, "model/conda.yaml")
        with open(conda_path) as f:
            conda_env = yaml.safe_load(f)

        COMPAT_OVERRIDES = {
            "scikit-learn": {"1.1.1": "1.2.2", "1.0.2": "1.2.2"},
        }

        model_reqs = {}
        for dep in conda_env.get("dependencies", []):
            if isinstance(dep, dict) and "pip" in dep:
                for pip_dep in dep["pip"]:
                    if "==" in pip_dep:
                        name, ver = pip_dep.split("==", 1)
                        model_reqs[name.strip().lower()] = ver.strip()

        issues = []
        for pkg, model_ver in model_reqs.items():
            try:
                installed_ver = pkg_version(pkg)
            except Exception:
                issues.append(f"  {pkg}=={model_ver} NOT INSTALLED")
                continue
            if installed_ver != model_ver:
                ok_vers = COMPAT_OVERRIDES.get(pkg, {})
                if model_ver in ok_vers and installed_ver == ok_vers[model_ver]:
                    pass  # Known override, fine
                else:
                    issues.append(f"  {pkg}: installed={installed_ver}, model needs={model_ver}")

        if issues:
            logger.warning(f"Dependency mismatches:\n" + "\n".join(issues))
    except Exception as e:
        logger.warning(f"Dep validation skipped: {e}")

validate_model_deps()

# ── Step 6: Run the Streamlit app ──
runpy.run_path("app.py", run_name="__main__")
