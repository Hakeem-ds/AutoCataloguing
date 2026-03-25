# transform.py

import pandas as pd


# ------------------------------------------------------------
# 1. Transform Wholly Amalgamated Sheet into Folder Mapping
# ------------------------------------------------------------
def transform_excel(df_standard):
    """
    Extracts Reference → Name pairs from hierarchical Wholly Amalgamated sheet.
    """
    if df_standard.empty:
        raise ValueError("Standard sheet is empty.")

    cols = ["Ref No"] + [f"Level {i}" for i in range(1, len(df_standard.columns))]
    df_standard.columns = cols

    ref_label_pairs = []

    for i in range(df_standard.shape[1] - 1):
        ref_col = df_standard.iloc[:, i]
        label_col = df_standard.iloc[:, i + 1]

        for ref, label in zip(ref_col, label_col):
            if isinstance(ref, str) and ref.strip().startswith("LT"):

                ref_label_pairs.append({
                    "Reference": ref.strip(),
                    "Name": str(label).strip() if label not in [None, "nan"] else ""
                })

    folder_mapping = pd.DataFrame(ref_label_pairs)
    folder_mapping["Name"] = folder_mapping["Name"].astype(str)
    folder_mapping = folder_mapping[folder_mapping["Name"].str.strip() != ""]

    return folder_mapping


# ------------------------------------------------------------
# 2. Process Split-Series Sheets (Colored Tabs)
# ------------------------------------------------------------
def process_colored_dfs(dfs):
    """
    Standardises all colored-sheet DataFrames into a single combined dataset
    containing: New SysID, Title, Date, Description
    """
    dfs_int = {}
    cols_of_interest = ["New SysID", "Title", "Date", "Description"]

    for key, df in dfs.items():

        df = df.copy()

        # Ensure first column is "New SysID" if unnamed
        if "New SysID" not in df.columns:
            df.rename(columns={df.columns[0]: "New SysID"}, inplace=True)

        # Keep relevant columns (some may not exist)
        available_cols = [c for c in cols_of_interest if c in df.columns]
        df = df[available_cols]

        # Convert everything to string for text processing
        df = df.astype(str)

        dfs_int[key] = df

    full_df = pd.concat(list(dfs_int.values()), ignore_index=True)

    return full_df