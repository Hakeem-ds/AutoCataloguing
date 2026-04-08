# core/feedback.py
import os
from typing import List, Dict, Any

import pandas as pd
from datetime import datetime

from .config import CONFIG
from .label_map import normalize_sys_id, normalise_to_taxonomy, load_folder_name_map


def _feedback_path() -> str:
    """Resolve feedback CSV path from CONFIG."""
    return CONFIG.get("feedback_csv") or "feedback.csv"


def _dedup_key(title: str, description: str, correct_sysid: str) -> str:
    """
    Deterministic key for deduplication.
    Lowered + stripped Title + Description + Correct SysID.
    """
    t = str(title).strip().lower()
    d = str(description).strip().lower()
    c = str(correct_sysid).strip().upper()
    return f"{t}||{d}||{c}"


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
      - Row already exists in feedback (dedup by lowered Title + Description + Correct SysID)

    Returns: summary counts.
    """
    path = _feedback_path()

    existing = load_feedback()
    new_df = pd.DataFrame(rows)

    if new_df.empty:
        return {"appended": 0, "skipped_empty_correct": 0, "skipped_equal_to_pred": 0, "skipped_duplicate": 0}

    # Ensure strings; do not normalize for storage (only for equality checks)
    new_df = new_df.map(lambda x: "" if x is None else str(x))

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

    # Normalise Correct SysID to taxonomy (zero-pad, file-level index)
    try:
        _fmap = load_folder_name_map()
        if _fmap and "Correct SysID" in filtered.columns and not filtered.empty:
            _taxonomy = set(_fmap.keys())
            filtered["Correct SysID"] = filtered["Correct SysID"].apply(
                lambda s: normalise_to_taxonomy(str(s).strip(), _taxonomy) if str(s).strip() else s
            )
    except Exception:
        pass

    if filtered.empty:
        return {
            "appended": 0,
            "skipped_empty_correct": skipped_empty_correct,
            "skipped_equal_to_pred": skipped_equal_to_pred,
            "skipped_duplicate": 0,
        }

    # ------------------------------------------------------------------
    # Dedup: remove rows that already exist in the feedback CSV
    # Uses lowered Title + Description + uppered Correct SysID as key
    # ------------------------------------------------------------------
    existing_keys = set()
    if not existing.empty:
        for _, er in existing.iterrows():
            existing_keys.add(_dedup_key(
                er.get("Title", ""),
                er.get("Description", ""),
                er.get("Correct SysID", ""),
            ))

    dedup_mask = []
    skipped_duplicate = 0
    for _, nr in filtered.iterrows():
        key = _dedup_key(nr["Title"], nr["Description"], nr["Correct SysID"])
        if key in existing_keys:
            skipped_duplicate += 1
            dedup_mask.append(False)
        else:
            existing_keys.add(key)   # prevent intra-batch dupes too
            dedup_mask.append(True)

    filtered = filtered[dedup_mask].copy()

    if filtered.empty:
        return {
            "appended": 0,
            "skipped_empty_correct": skipped_empty_correct,
            "skipped_equal_to_pred": skipped_equal_to_pred,
            "skipped_duplicate": skipped_duplicate,
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
        "skipped_duplicate": skipped_duplicate,
    }


def deduplicate_feedback() -> Dict[str, int]:
    """
    Deduplicate feedback.csv in-place.
    Uses lowered Title + Description + uppered Correct SysID as key.
    Returns summary: {"before": N, "after": N, "removed": N}.
    """
    path = _feedback_path()
    df = load_feedback()

    if df.empty:
        return {"before": 0, "after": 0, "removed": 0}

    before = len(df)

    # Build dedup key per row
    df["_dedup_key"] = df.apply(
        lambda r: _dedup_key(
            r.get("Title", ""),
            r.get("Description", ""),
            r.get("Correct SysID", ""),
        ),
        axis=1,
    )

    df = df.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])
    after = len(df)

    # Write back
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

    return {"before": before, "after": after, "removed": before - after}
