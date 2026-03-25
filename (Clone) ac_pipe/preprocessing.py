# preprocessing.py
import re

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None  # fall back if spaCy not available

def _lemmatize(text: str) -> str:
    if _nlp is None:
        return text
    doc = _nlp(text)
    return " ".join(t.lemma_ for t in doc)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"_x000D_|_x000A_", " ", text)
    text = text.replace("&amp;amp;", "&").replace("&amp;nbsp;", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    text = _lemmatize(text)
    return text

def clean_text_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' is missing in DataFrame.")
    return df[column_name].fillna("").astype(str).apply(clean_text)