import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from .label_map import normalize_sys_id, resolve_hierarchical_sys_id
from .text_cleaning import is_rubbish


# ------------------------------------------------------------
# Confidence Extraction (Compatible with sklearn Pipelines)
# ------------------------------------------------------------
def _confidence_from_model(model, X_row_df, predicted_label) -> Optional[float]:
    """
    Works with MLflow-loaded sklearn pipelines:
    - model.predict_proba (if calibrated)
    - model.decision_function (if margin only)
    - fallback softmax
    """

    try:
        pred_norm = normalize_sys_id(predicted_label)

        # --- 1) Calibrated Proba ---
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_row_df)[0]
            classes = getattr(model, "classes_", None)

            if classes is not None:
                probs_by_norm = {
                    normalize_sys_id(c): p for c, p in zip(classes, proba)
                }
                if pred_norm in probs_by_norm:
                    return float(probs_by_norm[pred_norm])
            return float(np.max(proba))

        # --- 2) Decision Function -> Softmax ---
        if hasattr(model, "decision_function"):
            df = model.decision_function(X_row_df)
            row = df[0] if np.ndim(df) > 1 else df

            exps = np.exp(row - np.max(row))
            soft = exps / np.sum(exps)

            classes = getattr(model, "classes_", None)
            if classes is not None:
                probs_by_norm = {
                    normalize_sys_id(c): p for c, p in zip(classes, soft)
                }
                if pred_norm in probs_by_norm:
                    return float(probs_by_norm[pred_norm])

            return float(np.max(soft))

    except Exception:
        return None

    return None


# ------------------------------------------------------------
# Single Prediction
# ------------------------------------------------------------
def predict_single(
    input_title: str,
    input_description: str,
    model,
    folder_name_map: Dict[str, str],
    valid_sysids: set[str],
    theme: str = "light",
) -> Tuple[str, str, str]:

    try:
        text = f"{input_title} {input_description}".strip()
        df = pd.DataFrame({"text": [text]})

        raw_pred = model.predict(df)[0]
        pred_sys_id = normalize_sys_id(raw_pred)

        canonical = resolve_hierarchical_sys_id(pred_sys_id, valid_sysids)
        if canonical:
            pred_sys_id = canonical

        conf_val = _confidence_from_model(model, df, raw_pred)
        conf_str = f"{conf_val * 100:.2f}%" if isinstance(conf_val, float) else "N/A"

        folder = folder_name_map.get(pred_sys_id, "Unknown")

        # -------------------------
        # UI Styling (unchanged)
        # -------------------------
        if theme == "dark":
            bg, fg, accent = "#23272f", "#f7fafc", "#4CAF50"
            fc, sc, cc = "#00bfff", "#00e676", "#ffb300"
        else:
            bg, fg, accent = "#f0f4f8", "#2a3f5f", "#4CAF50"
            fc, sc, cc = "#007acc", "#009966", "#cc6600"

        html = f"""
        <div style="background:{bg};color:{fg};padding:20px;border-radius:10px;">
            <h3 style="color:{accent};">📂 Prediction Result</h3>
            <b>Title:</b> {input_title}<br>
            <b>Description:</b> {input_description}<br><br>
            <b>🔢 SysID:</b> <span style="color:{sc};">{pred_sys_id}</span><br>
            <b>🗂 Folder:</b> <span style="color:{fc};">{folder}</span><br>
            <b>📈 Confidence:</b> <span style="color:{cc};">{conf_str}</span>
        </div>
        """

        return html, pred_sys_id, folder

    except Exception as e:
        return (f"<b style='color:red;'>Error: {str(e)}</b>", "", "")


# ------------------------------------------------------------
# Batch Prediction
# ------------------------------------------------------------
def predict_batch_df(
    df: pd.DataFrame,
    title_col: str,
    desc_col: str,
    model,
    folder_name_map: Dict[str, str],
    valid_sysids: set[str],
    true_sysid_col: Optional[str] = None,
) -> pd.DataFrame:

    df = df.dropna(subset=[title_col, desc_col], how="all").copy()
    if df.empty:
        return pd.DataFrame()

    idxs, texts = [], []
    for idx, r in df.iterrows():
        t = str(r.get(title_col, "") or "")
        d = str(r.get(desc_col, "") or "")
        if (t.strip() or d.strip()) and not (is_rubbish(t) and is_rubbish(d)):
            idxs.append(idx)
            texts.append(f"{t} {d}".strip())

    if not texts:
        return pd.DataFrame()

    df_text = pd.DataFrame({"text": texts})

    raw_preds = model.predict(df_text)

    out_rows = []
    for i, idx in enumerate(idxs):
        raw = raw_preds[i]
        pred = normalize_sys_id(raw)
        canonical = resolve_hierarchical_sys_id(pred, valid_sysids)
        if canonical:
            pred = canonical

        conf = _confidence_from_model(model, df_text.iloc[[i]], raw)
        conf_str = f"{conf * 100:.2f}%" if isinstance(conf, float) else "N/A"

        folder = folder_name_map.get(pred, "Unknown")
        t = str(df.loc[idx, title_col] or "")
        d = str(df.loc[idx, desc_col] or "")

        row = {
            "Source Index": idx,
            "Title": t,
            "Description": d,
            "Predicted SysID": pred,
            "Predicted Folder": folder,
            "Confidence": conf_str,
        }

        if true_sysid_col and true_sysid_col in df.columns:
            true_raw = df.loc[idx, true_sysid_col]
            true_norm = normalize_sys_id(true_raw)
            true_hier = resolve_hierarchical_sys_id(true_norm, valid_sysids)
            row["True SysID"] = true_hier or true_norm
            row["Match"] = (row["True SysID"] == pred)

        out_rows.append(row)

    return pd.DataFrame(out_rows)