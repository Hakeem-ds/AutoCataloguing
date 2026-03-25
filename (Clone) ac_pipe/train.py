# train.py
import mlflow
import mlflow.sklearn
from typing import Dict, Iterable, Optional, Tuple

import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mlflow.models.signature import infer_signature


def _flatten(prefix, d):
    return {f"{prefix}{k}": v for k, v in d.items()}


def _cap_cv_folds(y, requested_folds: int) -> int:
    """Cap CV folds to min(count per class) so CV never crashes."""
    counts = pd.Series(y).value_counts()
    min_per_class = int(counts.min()) if not counts.empty else 1
    return int(max(2, min(min_per_class, requested_folds)))


def train_model(
    model_type: str,
    train_df: pd.DataFrame,  # expects column "text"
    y_train,
    *,
    tfidf_params,
    calibration,
    svm=None,  # may be None
):
    """
    Returns: (pipeline, fit_time_seconds, metadata)
    """
    start = time.time()

    if model_type == "svm_linear_tfidf":

        # Ensure svm dict exists
        svm = svm or {}

        C = svm.get("C", 10.0)
        class_weight = svm.get("class_weight", "balanced")
        max_iter = svm.get("max_iter", 10000)

        calibrated = calibration.get("enabled", True)
        method = calibration.get("method", "sigmoid")
        cv_folds = calibration.get("cv_folds", 2)
        cv_shuffle = calibration.get("shuffle", True)
        cv_seed = calibration.get("seed", 42)

        pipeline = train_svm_tfidf(
            train_df=train_df,
            y_train=y_train,
            tfidf_params=tfidf_params,
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            calibrate=calibrated,
            calibration_method=method,
            cv_folds=cv_folds,
            cv_shuffle=cv_shuffle,
            cv_seed=cv_seed
        )

        meta = {
            "model_type": model_type,
            "calibrated": calibrated
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    fit_time = time.time() - start
    mlflow.log_metric("fit_time_sec", fit_time)

    return pipeline, fit_time, meta


def train_svm_tfidf(
    train_df: pd.DataFrame,   # expects column "text"
    y_train,
    *,
    tfidf_params=None,
    C=10.0,
    class_weight="balanced",
    max_iter=10000,
    calibrate=True,
    calibration_method="sigmoid",
    cv_folds=2,
    cv_shuffle=True,
    cv_seed=42,
):
    if tfidf_params is None:
        tfidf_params = {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "norm": "l2"
        }

    with mlflow.start_run(nested=True, run_name="training"):

        effective_folds = _cap_cv_folds(y_train, cv_folds)

        mlflow.log_params(_flatten("tfidf_", tfidf_params))
        mlflow.log_params({
            "svm_C": C,
            "svm_class_weight": class_weight,
            "svm_max_iter": max_iter,
            "calibrated": calibrate,
            "calibration_method": calibration_method,
            "cv_folds_requested": cv_folds,
            "cv_folds_effective": effective_folds,
        })

        base = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=cv_seed
        )

        if calibrate:
            cv = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=cv_shuffle,
                random_state=cv_seed
            )
            classifier = CalibratedClassifierCV(base, method=calibration_method, cv=cv)
        else:
            classifier = base

        text_selector = FunctionTransformer(lambda df: df["text"], validate=False)

        pipeline = Pipeline([
            ("select_text", text_selector),
            ("vectorizer", TfidfVectorizer(**tfidf_params)),
            ("classifier", classifier),
        ])

        pipeline.fit(train_df, y_train)

        input_example = pd.DataFrame({
            "text": [str(train_df["text"].iloc[0]) if len(train_df) else "example"]
        })

        signature = infer_signature(input_example, pipeline.predict(input_example))

        mlflow.sklearn.log_model(
            pipeline,
            "model",
            signature=signature,
            input_example=input_example
        )

        return pipeline