"""Named function used by re-logged sklearn pipelines (Python 3.11 compatible).
Replaces the original lambda df: df["text"] that was not portable across Python versions.
"""

def select_text_column(df):
    return df["text"]
