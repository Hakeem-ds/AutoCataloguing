# inference.py
import mlflow
import pandas as pd

def load_model(model_uri: str):
    pipeline = mlflow.sklearn.load_model(model_uri)
    return pipeline

def predict(model, texts):
    """
    texts: list[str] or pd.Series
    returns: dict with 'labels' and optional 'probs' if available
    """
    df = pd.DataFrame({"text": pd.Series(texts).astype(str)})
    try:
        preds = pipeline.predict(df)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    out = {"labels": preds}
    # try probs
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            out["probs"] = probs
    except Exception:
        pass
    return out