# training/evaluation.py — Model evaluation metrics
import mlflow
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from typing import Dict


def evaluate_model(model_pipeline, test_df, y_test,
                   train_df=None, y_train=None) -> Dict:
    """
    Evaluate a trained model with metrics suited for singleton-heavy datasets.

    Logs:
      1. Test accuracy + full classification report
      2. Macro/weighted precision, recall, F1
      3. Training data stats (samples, classes)
      4. Confidence distribution (margin-gap based)
      5. Cross-validation on training data
    """
    with mlflow.start_run(nested=True, run_name="evaluation"):

        metrics = {}

        # ── Training data stats ──
        if train_df is not None and y_train is not None:
            n_train = len(train_df)
            n_classes = int(y_train.nunique())
            mlflow.log_metric("n_training_samples", n_train)
            mlflow.log_metric("n_classes", n_classes)
            metrics["n_training_samples"] = n_train
            metrics["n_classes"] = n_classes

        # ── Prediction + timing ──
        t0 = time.time()
        preds = model_pipeline.predict(test_df)
        predict_time = time.time() - t0
        mlflow.log_metric("predict_time_sec", predict_time)
        mlflow.log_metric("num_test_samples", len(test_df))
        metrics["predict_time_sec"] = predict_time

        # ── Test accuracy + precision / recall / F1 ──
        if len(test_df) >= 5:
            acc = accuracy_score(y_test, preds)
            mlflow.log_metric("accuracy", float(acc))
            metrics["accuracy"] = acc

            try:
                report = classification_report(
                    y_test, preds, output_dict=True, zero_division=0
                )

                # F1 scores
                mlflow.log_metric("macro_f1", float(report["macro avg"]["f1-score"]))
                mlflow.log_metric("weighted_f1", float(report["weighted avg"]["f1-score"]))
                metrics["macro_f1"] = report["macro avg"]["f1-score"]
                metrics["weighted_f1"] = report["weighted avg"]["f1-score"]

                # Precision
                mlflow.log_metric("macro_precision", float(report["macro avg"]["precision"]))
                mlflow.log_metric("weighted_precision", float(report["weighted avg"]["precision"]))
                metrics["macro_precision"] = report["macro avg"]["precision"]
                metrics["weighted_precision"] = report["weighted avg"]["precision"]

                # Recall
                mlflow.log_metric("macro_recall", float(report["macro avg"]["recall"]))
                mlflow.log_metric("weighted_recall", float(report["weighted avg"]["recall"]))
                metrics["macro_recall"] = report["macro avg"]["recall"]
                metrics["weighted_recall"] = report["weighted avg"]["recall"]

                metrics["report"] = report

                with open("classification_report.json", "w") as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact("classification_report.json")
            except Exception as e:
                mlflow.set_tag("classification_report_error", str(e)[:200])
        else:
            mlflow.set_tag("eval_note", f"Test set too small ({len(test_df)} rows)")

        # ── Label coverage ──
        classes = getattr(model_pipeline, "classes_", None)
        if classes is not None:
            n_labels = len(classes)
            mlflow.log_metric("label_coverage", n_labels)
            metrics["label_coverage"] = n_labels

        # ── Confidence distribution ──
        _log_confidence_metrics(model_pipeline, test_df, metrics)

        # ── Cross-validation on training data ──
        if train_df is not None and y_train is not None and len(train_df) >= 10:
            _log_cv_score(model_pipeline, train_df, y_train, metrics)

        # ── Summary artifact ──
        loggable = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        with open("eval_summary.json", "w") as f:
            json.dump(loggable, f, indent=4)
        mlflow.log_artifact("eval_summary.json")

        return metrics


def _margin_gap_confidence(pipeline, test_df):
    """
    Compute margin-gap confidence: sigmoid(top_score - second_best_score).

    This gives:
      - High confidence (~1.0) when the model strongly prefers one class
      - Low confidence (~0.5) when two classes are close
    Works well regardless of number of classes (unlike softmax).
    """
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(test_df)
            # For calibrated models, use raw probabilities
            if proba.ndim == 1:
                return proba, "predict_proba"
            top2 = np.partition(proba, -2, axis=1)[:, -2:]
            top = top2[:, 1]  # highest
            second = top2[:, 0]  # second highest
            gaps = top - second
            return gaps, "predict_proba_gap"

        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(test_df)
            if scores.ndim == 1:
                return 1 / (1 + np.exp(-np.abs(scores))), "decision_function_sigmoid"

            # Multi-class: margin gap between top 2 classes
            top2 = np.partition(scores, -2, axis=1)[:, -2:]
            gaps = top2[:, 1] - top2[:, 0]  # gap between 1st and 2nd
            # Sigmoid to map gap -> [0, 1]
            conf = 1 / (1 + np.exp(-gaps))
            return conf, "decision_function_margin_gap"
    except Exception:
        pass
    return None, "unavailable"


def _log_confidence_metrics(pipeline, test_df, metrics: dict):
    """Log confidence distribution metrics and histogram."""

    conf, source = _margin_gap_confidence(pipeline, test_df)
    mlflow.set_tag("confidence_source", source)

    if conf is not None and len(conf) > 0:
        mlflow.log_metric("mean_confidence", float(np.mean(conf)))
        mlflow.log_metric("median_confidence", float(np.median(conf)))
        mlflow.log_metric("min_confidence", float(np.min(conf)))
        mlflow.log_metric("p10_confidence", float(np.percentile(conf, 10)))
        mlflow.log_metric("p90_confidence", float(np.percentile(conf, 90)))
        metrics["mean_confidence"] = float(np.mean(conf))
        metrics["median_confidence"] = float(np.median(conf))

        try:
            plt.figure(figsize=(8, 4))
            plt.hist(conf, bins=20, edgecolor="black", alpha=0.7)
            plt.title("Confidence Distribution (Margin Gap)")
            plt.xlabel("Confidence")
            plt.ylabel("Count")
            plt.axvline(np.mean(conf), color="red", linestyle="--", label=f"Mean: {np.mean(conf):.3f}")
            plt.legend()
            plt.savefig("confidence_hist.png", dpi=100, bbox_inches="tight")
            mlflow.log_artifact("confidence_hist.png")
            plt.close()
        except Exception:
            pass


def _log_cv_score(pipeline, train_df, y_train, metrics: dict):
    """Cross-validate on training data (handles small classes)."""
    try:
        from sklearn.base import clone

        label_counts = y_train.value_counts()
        min_count = label_counts.min()
        n_folds = min(3, min_count) if min_count >= 2 else 0

        if n_folds < 2:
            mlflow.set_tag("cv_note", "Too many singletons for cross-validation")
            return

        cloned = clone(pipeline)
        cv_scores = cross_val_score(
            cloned, train_df, y_train, cv=n_folds, scoring="accuracy"
        )
        mlflow.log_metric("cv_accuracy_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_accuracy_std", float(cv_scores.std()))
        mlflow.log_metric("cv_folds", n_folds)
        metrics["cv_accuracy_mean"] = float(cv_scores.mean())
        metrics["cv_accuracy_std"] = float(cv_scores.std())

    except Exception as e:
        mlflow.set_tag("cv_error", str(e)[:200])
