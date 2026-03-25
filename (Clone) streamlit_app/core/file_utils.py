# core/file_utils.py

import os
import joblib
import pandas as pd
from typing import Optional
import streamlit as st

from core.config import CONFIG


# ---------------------------------------------------------
#  EXISTING HELPERS (PRESERVED EXACTLY AS YOU HAD THEM)
# ---------------------------------------------------------

def ensure_file_exists(path: str, header: Optional[str] = None):
    """
    Ensures a CSV file exists. If not, creates it and writes the header.
    """
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            if header:
                f.write(header + "\n")


def append_csv_row(path: str, row_dict: dict, header: Optional[str] = None):
    """
    Appends a row to a CSV file. If the file doesn't exist, writes the header first.
    """
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = pd.DataFrame([row_dict])
        if not file_exists and header:
            writer.to_csv(f, index=False, header=True)
        else:
            writer.to_csv(f, index=False, header=False)


# ---------------------------------------------------------
#  FOLDER ↔ SYSID MAPPINGS
# ---------------------------------------------------------

def load_csv_or_excel(uploaded_file):
    """
    Load a CSV, XLSX, or XLS file into a pandas DataFrame.
    Returns None if the file cannot be read.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


def load_folder_mappings():
    """
    Loads folder_to_sysid.pkl and builds sysid_to_folder (case-insensitive).
    """
    folder_to_sysid = joblib.load(CONFIG["baseline_folder_to_sysid"])
    sysid_to_folder = {sysid.lower(): folder for folder, sysid in folder_to_sysid.items()}
    return folder_to_sysid, sysid_to_folder


# ---------------------------------------------------------
#  LABEL RESOLUTION + TRANSFORMATION
# ---------------------------------------------------------

def resolve_label(user_input, le_classes, folder_to_sysid):
    """
    Resolves a user-entered label into the canonical folder name.

    Accepts:
    - Folder Names (case-insensitive)
    - SysIDs / NO-codes (case-insensitive)

    Returns:
    - Canonical folder name if resolvable
    - Original input if unknown
    """
    raw = str(user_input).strip()
    raw_lower = raw.lower()

    # Model classes (folder names)
    le_lower = {c.lower(): c for c in le_classes}

    # SysID → Folder mapping
    sysid_lower = {s.lower(): f for f, s in folder_to_sysid.items()}

    # 1. Folder name
    if raw_lower in le_lower:
        return le_lower[raw_lower]

    # 2. SysID
    if raw_lower in sysid_lower:
        return sysid_lower[raw_lower]

    # 3. Unknown
    return raw


def transform_label(raw_label, label_map, le_classes, folder_to_sysid):
    """
    Converts raw user input into the canonical label used by the model.

    Steps:
    1. Resolve Folder/SysID → canonical folder name
    2. Apply label_map forward mapping (case-insensitive)
    3. Return resolved label if no mapping exists
    """
    resolved = resolve_label(raw_label, le_classes, folder_to_sysid)

    forward_lower = {k.lower(): v for k, v in label_map.get("forward", {}).items()}
    resolved_lower = resolved.lower()

    if resolved_lower in forward_lower:
        return forward_lower[resolved_lower]

    return resolved


def inverse_transform_label(folder_label, folder_to_sysid):
    """
    Converts a folder name back to its SysID (NO-code), case-insensitive.
    """
    folder_lower = str(folder_label).lower()
    folder_lookup = {f.lower(): sysid for f, sysid in folder_to_sysid.items()}
    return folder_lookup.get(folder_lower, folder_label)
