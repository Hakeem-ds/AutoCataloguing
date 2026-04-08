import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List

from .label_map import normalize_sys_id, resolve_hierarchical_sys_id, normalise_to_taxonomy, load_folder_name_map
from .text_cleaning import is_rubbish
from .config import CONFIG


# ------------------------------------------------------------
# Confidence Extraction (Margin-Gap based)
# ------------------------------------------------------------
def _confidence_from_model(model, X_row_df, predicted_label) -> Optional[float]:
    """
    Extract meaningful confidence from an sklearn pipeline.

    Strategy (in order of preference):
      1. predict_proba  — if calibrated (CalibratedClassifierCV)
      2. decision_function margin-gap — sigmoid of (top_score - 2nd_score)
    """
    try:
        pred_norm = normalize_sys_id(predicted_label)

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

        if hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_row_df)
            row = df_scores[0] if np.ndim(df_scores) > 1 else df_scores
            if np.ndim(row) == 0 or len(row) <= 1:
                return float(1 / (1 + np.exp(-abs(float(row)))))
            top2_idx = np.argpartition(row, -2)[-2:]
            top2_vals = row[top2_idx]
            gap = float(np.max(top2_vals)) - float(np.min(top2_vals))
            return float(1 / (1 + np.exp(-gap)))

    except Exception:
        return None
    return None


# ------------------------------------------------------------
# Normalise + resolve a SysID against taxonomy and folder map
# ------------------------------------------------------------
def _normalise_sysid(sysid: str, valid_sysids: set, folder_name_map: Dict[str, str]) -> str:
    """
    Full normalisation chain for any SysID:
      1. normalize_sys_id (format-level: uppercase, trim, hyphens)
      2. resolve_hierarchical_sys_id (walk-up to known trained label)
      3. normalise_to_taxonomy (zero-pad, typo fix, file-index strip)
    Returns the most canonical form available.
    """
    normed = normalize_sys_id(sysid)
    canonical = resolve_hierarchical_sys_id(normed, valid_sysids)
    if canonical:
        normed = canonical
    if folder_name_map:
        normed = normalise_to_taxonomy(normed, set(folder_name_map.keys()))
    return normed


# ------------------------------------------------------------
# Hierarchical Folder Name Resolution
# ------------------------------------------------------------
def _resolve_folder_name(sysid: str, folder_name_map: Dict[str, str]) -> str:
    """
    Get human-readable folder name for a SysID.
    Walks up the hierarchy when no exact match exists.
    Returns the first match, or the raw SysID as fallback.
    """
    if sysid in folder_name_map:
        return folder_name_map[sysid]
    parts = sysid.split("/")
    for depth in range(len(parts) - 1, 0, -1):
        prefix = "/".join(parts[:depth])
        if prefix in folder_name_map:
            return folder_name_map[prefix]
    return sysid


# ------------------------------------------------------------
# Confidence Band Classification
# ------------------------------------------------------------
def _classify_confidence(confidence: float, bands: Optional[Dict] = None) -> str:
    """
    Classify a confidence score into LOW / MODERATE / HIGH.
    Defaults from CONFIG["confidence_bands"]; overridable via *bands*.
    """
    if bands is None:
        bands = CONFIG.get(
            "confidence_bands", {"low_upper": 0.23, "high_lower": 0.92}
        )
    if confidence < bands["low_upper"]:
        return "LOW"
    elif confidence >= bands["high_lower"]:
        return "HIGH"
    return "MODERATE"


# ------------------------------------------------------------
# Related Context Helpers
# ------------------------------------------------------------
def _build_sibling_map(trained_labels) -> Dict[str, List[str]]:
    parent_map: Dict[str, List[str]] = {}
    for label in sorted(trained_labels):
        parts = label.split("/")
        parent = "/".join(parts[:-1]) if len(parts) >= 2 else label
        parent_map.setdefault(parent, []).append(label)
    return parent_map


def _classify_relationship(pred_sysid: str, alt_sysid: str) -> str:
    pred_parts = pred_sysid.split("/")
    alt_parts = alt_sysid.split("/")
    if (
        len(pred_parts) >= 2
        and len(alt_parts) >= 2
        and "/".join(pred_parts[:-1]) == "/".join(alt_parts[:-1])
    ):
        return "sibling"
    if (
        len(pred_parts) >= 2
        and len(alt_parts) >= 2
        and "/".join(pred_parts[:2]) == "/".join(alt_parts[:2])
    ):
        return "same branch"
    if pred_parts[0] == alt_parts[0]:
        return "same collection"
    return "different collection"


def get_related_context(
    model,
    text: str,
    folder_name_map: Dict[str, str],
    trained_labels: set,
    top_n: int = 5,
    confidence_bands: Optional[Dict] = None,
) -> Dict:
    """
    Return related context for a prediction to guide reviewer corrections.

    When confidence is LOW or MODERATE, also returns joined-index
    suggestions (parent-level folder neighbourhoods).

    All SysIDs are normalised to taxonomy (zero-pad, file-index strip)
    so folder names resolve correctly and are checked against the taxonomy.

    *confidence_bands* overrides CONFIG defaults when the user adjusts
    the threshold preset in the UI.
    """
    df_input = pd.DataFrame({"text": [text]})

    if not hasattr(model, "predict_proba"):
        return {"prediction": {}, "confidence_band": "",
                "related": [], "siblings": [],
                "collection": {}, "parent": {},
                "joined_suggestions": [], "review_hint": ""}

    proba = model.predict_proba(df_input)[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[::-1][: top_n]

    # --- Primary prediction (full normalisation) ---
    pred_sysid = _normalise_sysid(classes[top_indices[0]], trained_labels, folder_name_map)
    pred_conf = float(proba[top_indices[0]])
    pred_name = _resolve_folder_name(pred_sysid, folder_name_map)

    # --- Confidence band (uses override if provided) ---
    band = _classify_confidence(pred_conf, confidence_bands)

    # --- Related folders (normalised to taxonomy) ---
    related = []
    for idx in top_indices[1:]:
        alt_sysid = _normalise_sysid(classes[idx], trained_labels, folder_name_map)
        related.append({
            "sysid": alt_sysid,
            "folder_name": _resolve_folder_name(alt_sysid, folder_name_map),
            "confidence": float(proba[idx]),
            "relationship": _classify_relationship(pred_sysid, alt_sysid),
        })

    # --- Siblings (already taxonomy-level from trained_labels) ---
    sibling_map = _build_sibling_map(trained_labels)
    pred_parent = (
        "/".join(pred_sysid.split("/")[:-1])
        if "/" in pred_sysid
        else pred_sysid
    )
    siblings = [
        {
            "sysid": s,
            "folder_name": _resolve_folder_name(s, folder_name_map),
        }
        for s in sibling_map.get(pred_parent, [])
        if s != pred_sysid
    ][:8]

    # --- Collection context ---
    collection_id = pred_sysid.split("/")[0]
    collection_name = folder_name_map.get(collection_id, collection_id)

    # --- Joined-index suggestions (LOW + MODERATE only) ---
    joined_suggestions: List[Dict] = []
    if band in ("LOW", "MODERATE"):
        candidate_pool = min(top_n * 3, 15)
        pool_indices = np.argsort(proba)[::-1][:candidate_pool]
        parent_groups: Dict[str, Dict] = {}

        for idx in pool_indices:
            # Normalise each candidate to taxonomy
            sysid = _normalise_sysid(classes[idx], trained_labels, folder_name_map)
            conf = float(proba[idx])
            par = (
                "/".join(sysid.split("/")[:-1]) if "/" in sysid else sysid
            )
            if par not in parent_groups:
                parent_groups[par] = {
                    "combined_conf": 0.0,
                    "children": [],
                    "parent_name": _resolve_folder_name(par, folder_name_map),
                }
            parent_groups[par]["combined_conf"] += conf
            parent_groups[par]["children"].append({
                "sysid": sysid,
                "name": _resolve_folder_name(sysid, folder_name_map),
                "confidence": conf,
            })

        sorted_parents = sorted(
            parent_groups.items(),
            key=lambda x: x[1]["combined_conf"],
            reverse=True,
        )
        for par_sysid, info in sorted_parents[:3]:
            joined_suggestions.append({
                "parent_sysid": par_sysid,
                "parent_name": info["parent_name"],
                "combined_confidence": info["combined_conf"],
                "child_count": len(info["children"]),
                "children": info["children"][:5],
            })

    # --- Review hint ---
    if band == "HIGH":
        hint = f"High confidence ({pred_conf:.0%}). Likely correct."
    elif band == "MODERATE":
        hint = (
            f"Moderate confidence ({pred_conf:.0%}). "
            f"Check suggested neighbourhoods \u2014 correct class is often "
            f"in '{collection_name}'."
        )
    else:
        hint = (
            f"Low confidence ({pred_conf:.0%}). "
            f"Review suggested neighbourhoods and siblings carefully."
        )

    return {
        "prediction": {
            "sysid": pred_sysid,
            "folder_name": pred_name,
            "confidence": pred_conf,
        },
        "confidence_band": band,
        "related": related,
        "siblings": siblings,
        "collection": {"sysid": collection_id, "name": collection_name},
        "parent": {
            "sysid": pred_parent,
            "name": _resolve_folder_name(pred_parent, folder_name_map),
        },
        "joined_suggestions": joined_suggestions,
        "review_hint": hint,
    }


get_top_alternatives = get_related_context


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
    confidence_bands: Optional[Dict] = None,
) -> Tuple[str, str, str]:
    try:
        text = f"{input_title} {input_description}".strip()
        df = pd.DataFrame({"text": [text]})

        raw_pred = model.predict(df)[0]
        pred_sys_id = _normalise_sysid(raw_pred, valid_sysids, folder_name_map)

        conf_val = _confidence_from_model(model, df, raw_pred)
        conf_str = f"{conf_val * 100:.1f}%" if isinstance(conf_val, float) else "N/A"
        folder = _resolve_folder_name(pred_sys_id, folder_name_map)

        if isinstance(conf_val, float):
            band = _classify_confidence(conf_val, confidence_bands)
            if band == "HIGH":
                conf_color, conf_label = "#2e7d32", "High"
            elif band == "MODERATE":
                conf_color, conf_label = "#f57c00", "Moderate"
            else:
                conf_color, conf_label = "#c62828", "Low"
        else:
            conf_color, conf_label = "#757575", ""

        if theme == "dark":
            bg, fg, accent = "#23272f", "#f7fafc", "#4CAF50"
            fc, sc = "#00bfff", "#00e676"
        else:
            bg, fg, accent = "#f0f4f8", "#2a3f5f", "#4CAF50"
            fc, sc = "#007acc", "#009966"

        html = f"""
        <div style="background:{bg};color:{fg};padding:20px;border-radius:10px;">
            <h3 style="color:{accent};">\U0001f4c2 Prediction Result</h3>
            <b>Title:</b> {input_title}<br>
            <b>Description:</b> {input_description}<br><br>
            <b>\U0001f522 SysID:</b> <span style="color:{sc};">{pred_sys_id}</span><br>
            <b>\U0001f5c2 Folder:</b> <span style="color:{fc};">{folder}</span><br>
            <b>\U0001f4c8 Confidence:</b>
                <span style="color:{conf_color};font-weight:bold;">{conf_str}</span>
                <span style="color:{conf_color};font-size:0.85em;"> ({conf_label})</span>
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
    confidence_bands: Optional[Dict] = None,
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

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        all_proba = model.predict_proba(df_text)
        all_classes = model.classes_

    out_rows = []
    for i, idx in enumerate(idxs):
        raw = raw_preds[i]
        pred = _normalise_sysid(raw, valid_sysids, folder_name_map)

        conf = _confidence_from_model(model, df_text.iloc[[i]], raw)
        conf_str = f"{conf * 100:.1f}%" if isinstance(conf, float) else "N/A"
        band = _classify_confidence(conf, confidence_bands) if isinstance(conf, float) else ""
        folder = _resolve_folder_name(pred, folder_name_map)
        t = str(df.loc[idx, title_col] or "")
        d = str(df.loc[idx, desc_col] or "")

        row = {
            "Source Index": idx,
            "Title": t,
            "Description": d,
            "Predicted SysID": pred,
            "Predicted Folder": folder,
            "Confidence": conf_str,
            "Band": band,
        }

        if has_proba:
            top5_idx = np.argsort(all_proba[i])[::-1][1:6]
            related_summaries = []
            for rank, ti in enumerate(top5_idx, 2):
                # Normalise related SysIDs to taxonomy
                rel_sysid = _normalise_sysid(all_classes[ti], valid_sysids, folder_name_map)
                rel_folder = _resolve_folder_name(rel_sysid, folder_name_map)
                rel_conf = all_proba[i][ti]
                row[f"Related {rank} SysID"] = rel_sysid
                row[f"Related {rank} Folder"] = rel_folder
                row[f"Related {rank} Conf"] = f"{rel_conf * 100:.1f}%"
                # Summary: show both SysID and folder name for verification
                if rel_folder and rel_folder != rel_sysid:
                    related_summaries.append(f"{rel_sysid} \u2014 {rel_folder}")
                else:
                    related_summaries.append(rel_sysid)
            row["Related Context"] = " | ".join(
                s for s in related_summaries if s and not s.startswith(pred)
            )

        if true_sysid_col and true_sysid_col in df.columns:
            true_raw = df.loc[idx, true_sysid_col]
            true_norm = normalize_sys_id(true_raw)
            true_hier = resolve_hierarchical_sys_id(true_norm, valid_sysids)
            row["True SysID"] = true_hier or true_norm
            row["Match"] = (row["True SysID"] == pred)

        out_rows.append(row)

    return pd.DataFrame(out_rows)
