# core/label_map_cleaner.py

import pandas as pd

from core.label_map import load_label_map
from core.feedback import get_feedback_df
from core.training_data import get_training_data_df
from core.file_utils import resolve_label, load_folder_mappings


def _normalize_raw_label(label: str) -> str:
    return str(label).strip().lower()


def clean_label_map(le_classes: list[str]):
    """
    Returns a cleaned label_map and a diff DataFrame.

    - Normalizes raw labels (lowercase)
    - Drops mappings whose transformed label is not in le_classes
    - Rebuilds reverse map from forward
    """
    label_map = load_label_map()
    forward = label_map.get("forward", {})
    reverse = label_map.get("reverse", {})

    cleaned_forward: dict[str, str] = {}
    cleaned_reverse: dict[str, str] = {}

    # Build cleaned forward/reverse
    for raw, trans in forward.items():
        raw_norm = _normalize_raw_label(raw)
        if trans in le_classes:
            cleaned_forward[raw_norm] = trans
            cleaned_reverse[trans] = raw_norm

    # Build diff
    rows = []
    all_raw_keys = set(forward.keys()) | set(cleaned_forward.keys())
    for raw in sorted(all_raw_keys):
        before_trans = forward.get(raw)
        after_trans = cleaned_forward.get(_normalize_raw_label(raw))
        if before_trans == after_trans:
            status = "unchanged"
        elif before_trans is None and after_trans is not None:
            status = "added"
        elif before_trans is not None and after_trans is None:
            status = "removed"
        else:
            status = "changed"
        rows.append(
            {
                "raw_label_original": raw,
                "transformed_original": before_trans,
                "raw_label_cleaned": _normalize_raw_label(raw)
                if after_trans is not None
                else None,
                "transformed_cleaned": after_trans,
                "status": status,
            }
        )
    diff_df = pd.DataFrame(rows)

    # Update label_map structure
    label_map["forward"] = cleaned_forward
    label_map["reverse"] = cleaned_reverse
    if "seen_labels" not in label_map:
        label_map["seen_labels"] = []
    label_map["seen_labels"] = sorted(
        {*_normalize_raw_label(k) for k in cleaned_forward.keys()}
    )

    return label_map, diff_df


def clean_feedback(le_classes: list[str]):
    """
    Returns a cleaned feedback DataFrame:

    - Recomputes correct_label_transformed from correct_label_raw
      using resolve_label + folder_to_sysid.
    """
    df = get_feedback_df()
    if df.empty:
        return df

    folder_to_sysid, _ = load_folder_mappings()

    df = df.copy()
    df["correct_label_transformed"] = df["correct_label_raw"].apply(
        lambda x: resolve_label(x, le_classes, folder_to_sysid)
    )
    return df


def clean_training_data(le_classes: list[str]):
    """
    Returns a cleaned training_data DataFrame:

    - Recomputes label_transformed from label_raw
      using resolve_label + folder_to_sysid.
    """
    df = get_training_data_df()
    if df.empty:
        return df

    folder_to_sysid, _ = load_folder_mappings()

    df = df.copy()
    df["label_transformed"] = df["label_raw"].apply(
        lambda x: resolve_label(x, le_classes, folder_to_sysid)
    )
    return df
