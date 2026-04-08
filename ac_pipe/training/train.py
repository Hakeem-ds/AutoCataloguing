# training/train.py — Fresh SVM-TF-IDF training with robust calibration
from __future__ import annotations

import time
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from mlflow.models import infer_signature


# ── Pickle-safe text selector (referenced by cloudpickle in saved models) ──
def _select_text(df):
    """Extract 'text' column from DataFrame."""
    return df["text"]


# ── TF-IDF param sanitiser ──
_TFIDF_ALLOWED_KEYS = {
    "input", "encoding", "decode_error", "strip_accents", "lowercase",
    "preprocessor", "tokenizer", "stop_words", "token_pattern", "ngram_range",
    "analyzer", "max_df", "min_df", "max_features", "vocabulary", "binary",
    "dtype", "norm", "use_idf", "smooth_idf", "sublinear_tf",
}


def _sanitize_tfidf_params(tfidf_params: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the keys that TfidfVectorizer accepts, with type fixes."""
    if not isinstance(tfidf_params, dict):
        return {}
    cleaned = {k: v for k, v in tfidf_params.items() if k in _TFIDF_ALLOWED_KEYS}
    # YAML deserializes [1, 2] as a list; sklearn requires a tuple
    if "ngram_range" in cleaned and isinstance(cleaned["ngram_range"], list):
        cleaned["ngram_range"] = tuple(cleaned["ngram_range"])
    return cleaned


def _cap_cv_folds(y, requested_folds: int) -> int:
    """Cap CV folds to min(count per class) so CV never crashes."""
    counts = pd.Series(y).value_counts()
    min_per_class = int(counts.min()) if not counts.empty else 1
    return int(max(2, min(min_per_class, requested_folds)))


# ============================================================
# Public API
# ============================================================
def train_model(
    model_type: str,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    tfidf_params: Dict[str, Any],
    calibration: Dict[str, Any],
    svm: Dict[str, Any] | None = None,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Train a fresh model. Returns (pipeline, fit_time_sec, metadata)."""
    if model_type.lower() in {"svm_linear_tfidf", "svm_tfidf", "svm"}:
        return _train_svm_tfidf(
            train_df=train_df,
            y_train=y_train,
            tfidf_params=tfidf_params or {},
            calibration=calibration or {},
            svm=svm or {},
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _train_svm_tfidf(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    tfidf_params: Dict[str, Any],
    calibration: Dict[str, Any],
    svm: Dict[str, Any],
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """
    Train a fresh TF-IDF + LinearSVC pipeline with robust calibration.

    Calibration strategy:
      - If min_class_count >= cv_folds: use StratifiedKFold CV (standard)
      - If min_class_count < cv_folds but >= 2: cap folds to min_class_count
      - If min_class_count == 1 (singletons exist): use cv="prefit"
        → Train LinearSVC first, then calibrate on the training data
        → This avoids StratifiedKFold issues with singletons entirely

    Pipeline shape:
        Pipeline([
            ("select_text", FunctionTransformer(_select_text)),
            ("vectorizer",  TfidfVectorizer),
            ("classifier",  CalibratedClassifierCV | LinearSVC),
        ])
    """
    t0 = time.time()

    if "text" not in train_df.columns:
        raise ValueError("train_df must contain 'text' column")

    # ── TF-IDF ──
    clean_tfidf = _sanitize_tfidf_params(tfidf_params)

    # ── SVM ──
    C = svm.get("C", 10.0)
    class_weight = svm.get("class_weight", "balanced")
    max_iter = svm.get("max_iter", 10000)
    seed = svm.get("seed", 42)

    # ── Calibration config ──
    calibrate = calibration.get("enabled", False)
    method = calibration.get("method", "sigmoid")
    cv_folds = calibration.get("cv_folds", 3)
    cv_shuffle = calibration.get("shuffle", True)
    cv_seed = calibration.get("seed", 42)

    min_class_count = int(pd.Series(y_train).value_counts().min())
    mlflow.set_tag("min_class_count", str(min_class_count))
    mlflow.set_tag("calibration_requested", str(calibrate))

    # ── Build classifier with robust calibration ──
    base = LinearSVC(
        C=C, class_weight=class_weight, max_iter=max_iter, random_state=seed
    )

    if calibrate:
        if min_class_count >= 2:
            # Standard CV-based calibration
            effective_folds = _cap_cv_folds(y_train, cv_folds)
            mlflow.set_tag("effective_cv_folds", str(effective_folds))
            mlflow.set_tag("calibration_method", f"cv_{effective_folds}_fold")
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(
                n_splits=effective_folds, shuffle=cv_shuffle, random_state=cv_seed
            )
            classifier = CalibratedClassifierCV(base, method=method, cv=cv)
            mlflow.set_tag("calibration_used", "True")
            mlflow.set_tag("calibration_decision_reason",
                           f"CV calibration: min_class={min_class_count} >= 2")
        else:
            # Singleton classes exist → use prefit calibration
            # Train LinearSVC first, then calibrate the pre-fitted model
            mlflow.set_tag("effective_cv_folds", "prefit")
            mlflow.set_tag("calibration_method", "prefit")
            mlflow.set_tag("calibration_used", "True")
            mlflow.set_tag("calibration_decision_reason",
                           f"Prefit calibration: min_class={min_class_count} < 2, singletons present")
            classifier = "PREFIT"  # sentinel — handled below
    else:
        mlflow.set_tag("calibration_used", "False")
        mlflow.set_tag("effective_cv_folds", "0")
        mlflow.set_tag("calibration_decision_reason", "Calibration not requested")
        classifier = base

    mlflow.set_tag("training_mode", "fresh")

    # ── Build & fit pipeline ──
    if classifier == "PREFIT":
        # Two-phase: fit base, then wrap in prefit calibrator
        base_pipeline = Pipeline([
            ("select_text", FunctionTransformer(_select_text, validate=False)),
            ("vectorizer", TfidfVectorizer(**clean_tfidf)),
            ("classifier", base),
        ])
        base_pipeline.fit(train_df, y_train)

        # Extract fitted vectorizer + base classifier, wrap classifier in calibrator
        fitted_vec = base_pipeline.named_steps["vectorizer"]
        fitted_base = base_pipeline.named_steps["classifier"]

        calibrated_clf = CalibratedClassifierCV(fitted_base, method=method, cv="prefit")
        # Calibrate on the TF-IDF transformed training data
        X_tfidf = fitted_vec.transform(_select_text(train_df))
        calibrated_clf.fit(X_tfidf, y_train)

        # Rebuild pipeline with calibrated classifier
        pipeline = Pipeline([
            ("select_text", FunctionTransformer(_select_text, validate=False)),
            ("vectorizer", fitted_vec),
            ("classifier", calibrated_clf),
        ])
    else:
        pipeline = Pipeline([
            ("select_text", FunctionTransformer(_select_text, validate=False)),
            ("vectorizer", TfidfVectorizer(**clean_tfidf)),
            ("classifier", classifier),
        ])
        pipeline.fit(train_df, y_train)

    fit_time = time.time() - t0

    # ── Log model with signature ──
    input_example = pd.DataFrame({
        "text": [str(train_df["text"].iloc[0]) if len(train_df) else "example"]
    })
    signature = infer_signature(input_example, pipeline.predict(input_example))
    mlflow.sklearn.log_model(
        pipeline, "model", signature=signature, input_example=input_example
    )

    meta = {
        "model_type": "svm_linear_tfidf",
        "fit_time_sec": fit_time,
        "calibrated": calibrate,
        "n_classes": len(np.unique(y_train)),
        "n_samples": len(train_df),
    }

    return pipeline, fit_time, meta
