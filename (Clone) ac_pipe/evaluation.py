# evaluation.py
import mlflow
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score
)
from collections import Counter
from typing import Dict


def evaluate_model(model_pipeline, test_df, y_test, topk=(1, 3, 5)) -> Dict:
    with mlflow.start_run(nested=True, run_name="evaluation"):

        # --------------------------
        # Prediction + timing
        # --------------------------
        t0 = time.time()
        preds = model_pipeline.predict(test_df)
        predict_time = time.time() - t0
        mlflow.log_metric("predict_time_sec", predict_time)
        mlflow.log_metric("num_test_samples", len(test_df))

        # --------------------------
        # Top-K accuracy
        # --------------------------
        try:
            if hasattr(model_pipeline, "predict_proba"):
                proba = model_pipeline.predict_proba(test_df)
                for k in topk:
                    if k <= proba.shape[1]:
                        tk = float(top_k_accuracy_score(
                            y_test, proba, k=k, labels=model_pipeline.classes_
                        ))
                        mlflow.log_metric(f"top{k}_accuracy", tk)
            else:
                # Fallback for decision_function
                if hasattr(model_pipeline, "decision_function"):
                    scores = model_pipeline.decision_function(test_df)

                    if scores.ndim == 1:
                        scores = np.vstack([-scores, scores]).T

                    for k in topk:
                        topk_pred = scores.argsort(axis=1)[:, ::-1][:, :k]
                        classes = model_pipeline.classes_
                        y_true_idx = np.array([np.where(classes == y)[0][0] for y in y_test])
                        hit = (topk_pred == y_true_idx[:, None]).any(axis=1).mean()
                        mlflow.log_metric(f"top{k}_accuracy", float(hit))

        except Exception as e:
            mlflow.set_tag("topk_error", str(e))

        # --------------------------
        # Calibrated probabilities
        # --------------------------
        calibrated = hasattr(model_pipeline.named_steps.get("classifier", model_pipeline), "predict_proba")
        mlflow.log_param("model_calibrated", calibrated)

        calibrated_conf = None
        calibrated_probs = None

        if calibrated:
            try:
                calibrated_probs = model_pipeline.predict_proba(test_df)
                calibrated_conf = calibrated_probs.max(axis=1)
                mlflow.log_metric("mean_calibrated_conf", float(np.mean(calibrated_conf)))
            except Exception as e:
                mlflow.set_tag("predict_proba_error", str(e))
                calibrated = False

        # --------------------------
        # Raw margin scores
        # --------------------------
        raw_conf = None
        try:
            decision_scores = model_pipeline.decision_function(test_df)
            if len(getattr(decision_scores, "shape", ())) > 1:
                raw_conf = decision_scores.max(axis=1)
            else:
                raw_conf = decision_scores
            mlflow.log_metric("mean_raw_margin", float(np.mean(raw_conf)))
        except Exception:
            pass

        # --------------------------
        # Accuracy + Full Report
        # --------------------------
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", float(acc))

        report = classification_report(y_test, preds, output_dict=True)
        mlflow.log_metric("macro_f1", float(report["macro avg"]["f1-score"]))
        mlflow.log_metric("weighted_f1", float(report["weighted avg"]["f1-score"]))

        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("classification_report.json")

        # --------------------------
        # Histograms
        # --------------------------
        if calibrated_conf is not None:
            try:
                plt.figure()
                plt.hist(calibrated_conf, bins=20)
                plt.title("Calibrated Confidence Distribution")
                plt.savefig("calibrated_conf_hist.png")
                mlflow.log_artifact("calibrated_conf_hist.png")
                plt.close()
            except Exception:
                pass

        if raw_conf is not None:
            try:
                plt.figure()
                plt.hist(raw_conf, bins=20)
                plt.title("Raw Margin Score Distribution")
                plt.savefig("raw_conf_hist.png")
                mlflow.log_artifact("raw_conf_hist.png")
                plt.close()
            except Exception:
                pass

        # --------------------------
        # Final return
        # --------------------------
        return {
            "accuracy": acc,
            "predictions": preds,
            "conf_calibrated": calibrated_conf,
            "conf_raw": raw_conf,
            "report": report
        }