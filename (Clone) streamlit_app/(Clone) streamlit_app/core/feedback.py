# core/feedback.py
import os
from typing import List, Dict, Any

import pandas as pd
from datetime import datetime

from .config import CONFIG
from .label_map import normalize_sys_id


def _feedback_path() -> str:
    """Resolve feedback CSV path from CONFIG."""
    return CONFIG.get("feedback_csv") or "feedback.csv"


def load_feedback() -> pd.DataFrame:
    """
    Load the feedback CSV if present, else return an empty DataFrame.
    Non-destructive: returns exactly what's on disk (no mutation).
    """
    path = _feedback_path()
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return df
    except Exception:
        return pd.DataFrame()


def append_feedback_rows(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Append feedback rows to the feedback CSV on disk.

    Stores only relevant fields:
      - Title
      - Description
      - Predicted SysID
      - Correct SysID
      - Model Version
      - Timestamp

    Skips rows where:
      - Correct SysID empty/missing
      - Normalized Correct SysID == normalized Predicted SysID

    Returns: summary counts.
    """
    path = _feedback_path()

    existing = load_feedback()
    new_df = pd.DataFrame(rows)

    if new_df.empty:
        return {"appended": 0, "skipped_empty_correct": 0, "skipped_equal_to_pred": 0}

    # Ensure strings; do not normalize for storage (only for equality checks)
    new_df = new_df.applymap(lambda x: "" if x is None else str(x))

    # Add Timestamp if missing
    if "Timestamp" not in new_df.columns:
        new_df["Timestamp"] = datetime.utcnow().isoformat()

    # Keep only relevant columns (others are discarded by design)
    keep_cols = [
        "Title",
        "Description",
        "Predicted SysID",
        "Correct SysID",
        "Model Version",
        "Timestamp",
    ]
    for col in keep_cols:
        if col not in new_df.columns:
            new_df[col] = ""

    new_df = new_df[keep_cols].copy()

    skipped_empty_correct = 0
    skipped_equal_to_pred = 0

    def _valid(r: pd.Series) -> bool:
        nonlocal skipped_empty_correct, skipped_equal_to_pred
        corr = r.get("Correct SysID", "")
        if not str(corr).strip():
            skipped_empty_correct += 1
            return False
        pred = r.get("Predicted SysID", "")
        if normalize_sys_id(corr) == normalize_sys_id(pred):
            skipped_equal_to_pred += 1
            return False
        return True

    mask_valid = new_df.apply(_valid, axis=1)
    filtered = new_df[mask_valid].copy()

    if filtered.empty:
        return {
            "appended": 0,
            "skipped_empty_correct": skipped_empty_correct,
            "skipped_equal_to_pred": skipped_equal_to_pred,
        }

    # Union columns across existing + new (but both should be keep_cols)
    if existing.empty:
        out_df = filtered
    else:
        all_cols = list(dict.fromkeys(existing.columns.tolist() + filtered.columns.tolist()))
        out_df = pd.concat(
            [existing.reindex(columns=all_cols, fill_value=""), filtered.reindex(columns=all_cols, fill_value="")],
            ignore_index=True
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, path)

    return {
        "appended": len(filtered),
        "skipped_empty_correct": skipped_empty_correct,
        "skipped_equal_to_pred": skipped_equal_to_pred,
    }