# core/training_logic.py
import os
import hashlib
from typing import List, Dict, Any

import pandas as pd
from datetime import datetime

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import CONFIG
from .feedback import load_feedback
from .label_map import normalize_sys_id, resolve_hierarchical_sys_id
from .model_loader import load_current_model


TRAIN_COLS = ["Title", "Description", "SysID", "RowID"]


def _row_id(title: str, description: str, sys_id: str) -> str:
    """Deterministic row ID for deduplication."""
    t = (title or "").strip().lower()
    d = (description or "").strip().lower()
    s = (sys_id or "").strip().upper()
    payload = f"{t}||{d}||{s}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# ============================================================
# 1) Build training data from feedback (UNCHANGED FUNCTIONALITY)
# ============================================================

def build_training_from_feedback(min_confidence: float = 0.0) -> pd.DataFrame:
    """
    Produce deduplicated training CSV from feedback.
    Uses Correct SysID if present; fallback to Predicted SysID.
    """
    fb = load_feedback()
    if fb.empty:
        return pd.DataFrame(columns=TRAIN_COLS)

    # Load trained labels
    _, _, _, valid_sysids, _ = load_current_model()

    rows = []
    for _, r in fb.iterrows():
        title = str(r.get("Title", "") or "")
        desc = str(r.get("Description", "") or "")

        corr = normalize_sys_id(r.get("Correct SysID", ""))
        pred = normalize_sys_id(r.get("Predicted SysID", ""))

        label = corr if corr else pred
        label = resolve_hierarchical_sys_id(label, valid_sysids) or label

        if not title.strip() and not desc.strip():
            continue
        if not label:
            continue

        rowid = _row_id(title, desc, label)
        rows.append({"Title": title, "Description": desc, "SysID": label, "RowID": rowid})

    df = pd.DataFrame(rows)

    # Deduplicate
    if not df.empty:
        df.sort_values(by=["RowID"], inplace=True)
        df = df.drop_duplicates(subset=["RowID"]).reset_index(drop=True)

    # Persist
    df.to_csv(CONFIG["training_data_csv"], index=False)
    return df


# ============================================================
# 2) Append external training rows (UNCHANGED)
# ============================================================

def append_training_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Append external training rows with deduplication.
    Expected keys: Title, Description, SysID
    """
    _, _, _, valid_sysids, _ = load_current_model()

    cleaned = []
    for r in rows:
        title = str(r.get("Title", "") or "")
        desc = str(r.get("Description", "") or "")
        sid = normalize_sys_id(r.get("SysID", ""))
        sid = resolve_hierarchical_sys_id(sid, valid_sysids) or sid
        if not sid:
            continue

        rowid = _row_id(title, desc, sid)
        cleaned.append({"Title": title, "Description": desc, "SysID": sid, "RowID": rowid})

    new_df = pd.DataFrame(cleaned)

    path = CONFIG["training_data_csv"]
    if os.path.exists(path):
        old = pd.read_csv(path)
    else:
        old = pd.DataFrame(columns=TRAIN_COLS)

    merged = pd.concat([old, new_df], ignore_index=True)
    if not merged.empty:
        merged.sort_values(by=["RowID"], inplace=True)
        merged = merged.drop_duplicates(subset=["RowID"]).reset_index(drop=True)

    merged.to_csv(path, index=False)
    return merged


# ============================================================
# 3) FULL RETRAIN LOGIC (Calibrated LinearSVC with prefit calibration)
# ============================================================

def run_retraining() -> Dict[str, Any]:
    """
    Full retraining pipeline:
    - Reads training_data.csv
    - (Optional) Safe validation split for reporting
    - TF-IDF fit
    - Fit base LinearSVC on training
    - Calibrate with CalibratedClassifierCV(cv='prefit') on a small calibration subset if possible,
      else on the same training data (warning scenario)
    - Save versioned artifacts
    - Return metadata for UI
    """
    tpath = CONFIG["training_data_csv"]
    if not os.path.exists(tpath):
        raise FileNotFoundError("training_data.csv does not exist.")

    tdf = pd.read_csv(tpath)
    if tdf.empty:
        raise ValueError("Training data is empty.")

    # Build text
    tdf["Title"] = tdf["Title"].fillna("")
    tdf["Description"] = tdf["Description"].fillna("")
    tdf["__text__"] = tdf["Title"] + " " + tdf["Description"]

    texts_all = tdf["__text__"].tolist()
    labels_all = tdf["SysID"].astype(str).tolist()

    unique_classes = sorted(set(labels_all))
    num_classes = len(unique_classes)
    n_samples = len(labels_all)

    if num_classes < 2:
        raise ValueError("Need at least 2 distinct SysIDs to train a classifier.")

    # -----------------------------
    # Optional validation split (for reporting only)
    # -----------------------------
    import math
    val_frac = 0.15
    val_size = math.floor(val_frac * n_samples)

    can_val = (
        all(labels_all.count(c) >= 2 for c in unique_classes)
        and n_samples >= 2 * num_classes
        and val_size >= num_classes
    )

    if can_val:
        X_train_all, X_val, y_train_all, y_val = train_test_split(
            texts_all, labels_all,
            test_size=val_frac, random_state=42, stratify=labels_all
        )
        had_validation = True
    else:
        X_train_all, y_train_all = texts_all, labels_all
        X_val, y_val = [], []
        had_validation = False

    # -----------------------------
    # Calibration split (independent of validation)
    # Try to carve out a small calibration subset from training; else calibrate on full training
    # -----------------------------
    cal_frac = 0.2
    cal_size = math.floor(cal_frac * len(y_train_all))

    can_calibrate_on_subset = (
        len(set(y_train_all)) >= 2 and
        cal_size >= num_classes and
        all(y_train_all.count(c) >= 2 for c in set(y_train_all))
    )

    if can_calibrate_on_subset:
        X_base, X_cal, y_base, y_cal = train_test_split(
            X_train_all, y_train_all,
            test_size=cal_frac, random_state=123, stratify=y_train_all
        )
        calib_note = f"calibrated on held-out subset (n={len(y_cal)})"
    else:
        # Calibrate on the same training set (still valid with cv='prefit')
        X_base, y_base = X_train_all, y_train_all
        X_cal, y_cal = X_train_all, y_train_all
        calib_note = "calibrated on training set (no separate calibration split)"

    # -----------------------------
    # Vectorize on the base training split
    # -----------------------------
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_base)

    # -----------------------------
    # Fit base LinearSVC (same as original)
    # -----------------------------
    base_svc = LinearSVC(
        C=10,
        class_weight="balanced",
        random_state=42,
        max_iter=10000,
    )
    base_svc.fit(Xtr, y_base)

    # -----------------------------
    # Calibrate with prefit
    # -----------------------------
    clf = CalibratedClassifierCV(base_svc, cv="prefit", method="sigmoid")
    Xcal = vec.transform(X_cal)
    clf.fit(Xcal, y_cal)

    # -----------------------------
    # Validation metrics (if we had a val split)
    # -----------------------------
    acc = None
    f1 = None
    if had_validation and len(X_val) > 0:
        Xva = vec.transform(X_val)
        preds_val = clf.predict(Xva)
        acc = accuracy_score(y_val, preds_val)
        f1 = f1_score(y_val, preds_val, average="macro")

    # -----------------------------
    # Save versioned artifacts
    # -----------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version = f"svm_v{timestamp}"

    out_dir = os.path.join(CONFIG["versioned_models_dir"], version)
    os.makedirs(out_dir, exist_ok=True)

    vec_path = os.path.join(out_dir, "tfidf_vectorizer.pkl")
    mdl_path = os.path.join(out_dir, "svm_model.pkl")

    import joblib
    joblib.dump(vec, vec_path)
    joblib.dump(clf, mdl_path)

    # -----------------------------
    # Registry update
    # -----------------------------
    from .model_registry import get_model_registry, save_model_registry

    reg = get_model_registry()
    reg[version] = {
        "vectorizer_path": vec_path,
        "model_path": mdl_path,
        "folder_name_map": CONFIG.get("folder_name_map"),
        "folder_mapping_csv": CONFIG.get("folder_mapping_csv"),
        "trained_on": datetime.utcnow().isoformat(),
        "notes": (
            "Calibrated LinearSVC (C=10, sigmoid, prefit); "
            f"samples={n_samples}, classes={num_classes}, "
            f"{'val acc=' + f'{acc*100:.2f}%, f1=' + f'{f1:.3f}' if acc is not None else 'no validation'}, "
            f"{calib_note}"
        ),
    }
    save_model_registry(reg)

    return {
        "version": version,
        "samples": n_samples,
        "classes": num_classes,
        "accuracy": acc,
        "macro_f1": f1,
        "had_validation": had_validation,
        "vectorizer_path": vec_path,
        "model_path": mdl_path,
        "calibration": calib_note,
    }