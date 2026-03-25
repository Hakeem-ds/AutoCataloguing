import os
import json
import joblib
from typing import Dict, Any, Tuple, Optional, Set, List

import pandas as pd
import mlflow

from .config import CONFIG
from .label_map import normalize_sys_id

# ---------------------------------------------------------------------
# Module constants / helpers
# ---------------------------------------------------------------------

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEBUG = False

def _dbg(*args):
    if _DEBUG:
        print("[model_registry]", *args)

def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _file_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path)

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_load_folder_name_map(path: Optional[str]) -> Dict[str, str]:
    if not _file_exists(path):
        return {}
    try:
        data = _load_json(path)
        return {normalize_sys_id(k): v for k, v in data.items()}
    except Exception:
        return {}

def _safe_build_valid_sysids(folder_name_map: Dict[str, str],
                             folder_mapping_csv: Optional[str]) -> Set[str]:
    if folder_name_map:
        return set(folder_name_map.keys())

    if _file_exists(folder_mapping_csv):
        try:
            df = pd.read_csv(folder_mapping_csv)
            col = None
            for c in ["sys_id", "sysid", "id", "folder_id"]:
                if c in df.columns:
                    col = c
                    break
            if col:
                return set(normalize_sys_id(x) for x in df[col].dropna().astype(str).tolist())
        except Exception:
            pass

    return set()

def _to_abs(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(_APP_ROOT, path))

def _assert_vectorizer_fitted(vectorizer: Any, vec_path: str):
    has_vocab = hasattr(vectorizer, "vocabulary_")
    has_idf = hasattr(vectorizer, "idf_") or hasattr(getattr(vectorizer, "_tfidf", object()), "idf_")

    if not (has_vocab and has_idf):
        raise ValueError(
            f"Unversioned Vectorizer at '{vec_path}' is not fitted. "
            "Ensure fit_transform() happened before saving."
        )

def _extract_trained_labels(model: Any) -> Set[str]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return set(normalize_sys_id(c) for c in classes)

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        for _, step in reversed(list(named_steps.items())):
            step_classes = getattr(step, "classes_", None)
            if step_classes is not None:
                return set(normalize_sys_id(c) for c in step_classes)

    return set()

# ---------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------

def get_model_registry() -> dict:
    path = CONFIG["model_registry_json"]
    if not os.path.exists(path):
        _atomic_write(path, {})
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_model_registry(registry: dict):
    _atomic_write(CONFIG["model_registry_json"], registry)

def list_model_versions() -> List[str]:
    reg = get_model_registry()
    versions = []

    if reg:
        items = [(k, v) for k, v in reg.items() if isinstance(v, dict)]
        items.sort(key=lambda kv: kv[1].get("trained_on") or "", reverse=True)
        versions = [k for k, _ in items]

    if _file_exists(CONFIG.get("vectorizer")) and _file_exists(CONFIG.get("svm_model")):
        if "unversioned" not in versions:
            versions.insert(0, "unversioned")

    return versions

def get_current_model_version() -> Optional[str]:
    versions = list_model_versions()
    if versions:
        return versions[0]
    if _file_exists(CONFIG.get("vectorizer")) and _file_exists(CONFIG.get("svm_model")):
        return "unversioned"
    return None

# ---------------------------------------------------------------------
# Unversioned loader
# ---------------------------------------------------------------------

def _load_unversioned_model():
    vec_path = _to_abs(CONFIG.get("vectorizer"))
    mdl_path = _to_abs(CONFIG.get("svm_model"))
    fmap_path = _to_abs(CONFIG.get("folder_name_map"))
    fmap_csv = _to_abs(CONFIG.get("folder_mapping_csv"))

    if not (_file_exists(vec_path) and _file_exists(mdl_path)):
        raise FileNotFoundError(f"Missing unversioned artifacts: {vec_path}, {mdl_path}")

    vectorizer = joblib.load(vec_path)
    model = joblib.load(mdl_path)

    _assert_vectorizer_fitted(vectorizer, vec_path)

    folder_name_map = _safe_load_folder_name_map(fmap_path)

    valid_sysids = _extract_trained_labels(model)
    if not valid_sysids:
        valid_sysids = _safe_build_valid_sysids(folder_name_map, fmap_csv)

    meta = {
        "version": "unversioned",
        "vectorizer_path": vec_path,
        "model_path": mdl_path,
        "folder_name_map": fmap_path,
        "folder_mapping_csv": fmap_csv,
        "trained_on": None,
        "notes": None,
        "classes": list(valid_sysids),
    }

    return vectorizer, model, folder_name_map, valid_sysids, meta

# ---------------------------------------------------------------------
# Versioned MLflow loader  (⭐ NEW)
# ---------------------------------------------------------------------

def load_model_by_version(version: str) -> Tuple[Any, Any, Dict[str, str], Set[str], Dict[str, Any]]:
    """
    Returns:
        vectorizer=None,
        model_pipeline,
        folder_name_map,
        valid_sysids,
        meta
    """
    _dbg("loading version", version)

    # -------------------------
    # 1) Legacy unversioned
    # -------------------------
    if version == "unversioned":
        return _load_unversioned_model()

    # -------------------------
    # 2) MLflow versioned
    # -------------------------
    registry = get_model_registry()
    if version not in registry:
        raise KeyError(f"Model version '{version}' not found in registry.")

    entry = registry[version]
    model_uri = entry.get("mlflow_model_uri")

    if not model_uri:
        raise KeyError(
            f"Model version '{version}' is missing 'mlflow_model_uri'. "
            "Ensure training pipeline logs this into the registry."
        )

    # Load sklearn pipeline
    try:
        model_pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load MLflow model from URI {model_uri}: {e}")

    # Folder name map (taxonomy)
    fmap_path = _to_abs(entry.get("folder_name_map") or CONFIG.get("folder_name_map"))
    folder_name_map = _safe_load_folder_name_map(fmap_path)

    # Extract trained labels
    valid_sysids = _extract_trained_labels(model_pipeline)

    meta = {
        "version": version,
        "mlflow_model_uri": model_uri,
        "folder_name_map": fmap_path,
        "trained_on": entry.get("trained_on"),
        "notes": entry.get("notes"),
        "classes": list(valid_sysids),
    }

    # NEW standard interface: vectorizer=None, model=model_pipeline
    return None, model_pipeline, folder_name_map, valid_sysids, meta