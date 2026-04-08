# core/file_utils.py

import os
import json
from typing import Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
#  Shared Utilities
# ---------------------------------------------------------

def atomic_write(path: str, data: dict):
    """Write a JSON dict to *path* atomically via a temp file + os.replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def safe_rerun():
    """Trigger a Streamlit rerun (compatible across Streamlit versions)."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ---------------------------------------------------------
#  CSV / File Helpers
# ---------------------------------------------------------

def ensure_file_exists(path: str, header: Optional[str] = None):
    """Ensures a CSV file exists. If not, creates it and writes the header."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            if header:
                f.write(header + "\n")


def append_csv_row(path: str, row_dict: dict, header: Optional[str] = None):
    """Appends a row to a CSV file. If the file doesn't exist, writes the header first."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = pd.DataFrame([row_dict])
        if not file_exists and header:
            writer.to_csv(f, index=False, header=True)
        else:
            writer.to_csv(f, index=False, header=False)


def load_csv_or_excel(uploaded_file):
    """Load a CSV, XLSX, or XLS file into a pandas DataFrame.
    Returns None if the file cannot be read."""
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
