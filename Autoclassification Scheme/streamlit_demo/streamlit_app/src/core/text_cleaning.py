import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)


def is_rubbish(text: str) -> bool:
    if not isinstance(text, str) or len(text.strip()) < 8:
        return True
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) > 0 and max([chars.count(c) for c in set(chars)]) / len(chars) > 0.7:
        return True
    alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
    if alpha_ratio < 0.5:
        return True
    words = text.lower().split()
    if len(words) > 2 and max([words.count(w) for w in set(words)]) / len(words) > 0.7:
        return True
    return False
