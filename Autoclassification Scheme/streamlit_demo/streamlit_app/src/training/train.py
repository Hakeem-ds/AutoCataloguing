# training/train.py — Deserialization shim
#
# The sklearn Pipeline contains FunctionTransformer(_select_text).
# When MLflow deserializes the model, Python must find this function
# at its original module path: training.train._select_text
#
# This file exists ONLY for that purpose. The canonical training code
# lives in ac_pipe/training/train.py. Do not add training logic here.

def _select_text(df):
    """Extract 'text' column from DataFrame — used by FunctionTransformer in the pipeline."""
    return df["text"]
