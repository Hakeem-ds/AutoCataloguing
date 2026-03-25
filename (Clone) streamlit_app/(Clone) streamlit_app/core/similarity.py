# core/similarity.py
from typing import List, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def _ensure_vectorizer(obj: Any):
    """Accept either a TfidfVectorizer or a sklearn Pipeline and return the vectorizer."""
    vec = getattr(obj, "transform", None) and obj  # if it looks like a vectorizer
    if vec and hasattr(vec, "vocabulary_"):
        return obj
    # else, try pipeline
    named = getattr(obj, "named_steps", None)
    if isinstance(named, dict) and "vectorizer" in named:
        return named["vectorizer"]
    raise ValueError("Could not extract a TfidfVectorizer. Pass a vectorizer or a Pipeline with 'vectorizer' step.")

def compute_tfidf_cosine_similarity(docs: List[str], vectorizer_or_model) -> np.ndarray:
    vec = _ensure_vectorizer(vectorizer_or_model)
    tfidf_matrix = vec.transform(docs)
    return cosine_similarity(tfidf_matrix)

def compute_keyword_weighted_similarity(docs: List[str], vectorizer_or_model, titles_only: Optional[list[str]] = None) -> np.ndarray:
    vec = _ensure_vectorizer(vectorizer_or_model)
    tfidf_matrix = vec.transform(titles_only) if titles_only is not None else vec.transform(docs)
    weights = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
    sim_matrix = cosine_similarity(tfidf_matrix)
    sim_matrix = sim_matrix * weights[:, None]
    min_val = np.min(sim_matrix)
    max_val = np.max(sim_matrix)
    if max_val > min_val:
        sim_matrix = (sim_matrix - min_val) / (max_val - min_val)
    sim_matrix = sim_matrix * 100
    return sim_matrix

def compute_hybrid_similarity(docs: List[str], vectorizer_or_model, titles_only: Optional[list[str]] = None, alpha: float = 0.6) -> np.ndarray:
    tfidf_sim = compute_tfidf_cosine_similarity(docs, vectorizer_or_model)
    kw_sim = compute_keyword_weighted_similarity(docs, vectorizer_or_model, titles_only=titles_only) / 100.0
    hybrid_sim = alpha * tfidf_sim + (1 - alpha) * kw_sim
    return hybrid_sim


def get_similarity_color(score: float) -> str:
    if score <= 25:
        return "#e53935"
    elif score <= 50:
        return "#fb8c00"
    elif score <= 75:
        return "#fdd835"
    else:
        return "#43a047"


def group_similar_docs(sim_matrix: np.ndarray, threshold: float = 50) -> list[list[int]]:
    n = sim_matrix.shape[0]
    groups = []
    visited = set()
    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(n):
            if j != i and sim_matrix[i, j] >= threshold and j not in visited:
                group.append(j)
                visited.add(j)
        if group:
            groups.append(group)
    return groups


def cluster_documents(sim_matrix: np.ndarray, n_clusters: int = 3, method: str = "agglomerative") -> np.ndarray:
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(sim_matrix)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
        distance_matrix = 1 - (sim_matrix / 100.0)
        labels = clustering.fit_predict(distance_matrix)
    return labels
