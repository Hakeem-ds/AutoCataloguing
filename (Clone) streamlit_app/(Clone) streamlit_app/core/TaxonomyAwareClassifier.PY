import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class TaxonomyAwareClassifier:
    """
    Version 1 — SAFE WRAPPER
    -------------------------
    This class encapsulates:
      - SVM text classifier pipeline
      - Taxonomy-based retrieval index
      - Confidence thresholding
      - Prediction with fallback suggestions

    IMPORTANT:
        This version does NOT break your MVP.
        It does NOT change your training or prediction code.
        You can integrate it one method at a time.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        max_features: int = 30000,
        lowercase: bool = True,
        stop_words: str | None = "english"
    ):
        self.conf_threshold = confidence_threshold

        # Main classifier pipeline
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None

        # Retrieval (taxonomy) engine
        self.retr_vec: TfidfVectorizer | None = None
        self.tax_mat: sparse.csr_matrix | None = None
        self.tax_sysids: List[str] | None = None
        self.tax_labels: List[str] | None = None

        # TF-IDF hyperparameters
        self.tfidf_kwargs = dict(
            lowercase=lowercase,
            stop_words=stop_words,
            max_features=max_features
        )


    # ------------------------------------------------------
    # 1. BUILD CLASSIFIER PIPELINE
    # ------------------------------------------------------
    @staticmethod
    def _build_classifier_pipeline(tfidf_kwargs):
        return Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", CalibratedClassifierCV(
                base_estimator=LinearSVC(),
                cv=5,
                method="sigmoid"
            ))
        ])


    # ------------------------------------------------------
    # 2. TRAIN CLASSIFIER
    # ------------------------------------------------------
    def fit_classifier(self, texts: List[str], labels: List[str]):
        """
        Fit SVM classifier using your training data.
        Does NOT affect taxonomy or retrieval yet.
        """
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        self.pipeline = self._build_classifier_pipeline(self.tfidf_kwargs)
        self.pipeline.fit(texts, y)


    # ------------------------------------------------------
    # 3. BUILD TAXONOMY RETRIEVAL INDEX
    # ------------------------------------------------------
    def fit_taxonomy(self, sysids: List[str], names_or_desc: List[str]):
        """
        Build retrieval index for taxonomy guidance and search.
        """
        self.retr_vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=30000
        )
        self.tax_mat = self.retr_vec.fit_transform(names_or_desc).tocsr()
        self.tax_sysids = sysids
        self.tax_labels = names_or_desc


    # ------------------------------------------------------
    # 4. PREDICT + CONFIDENCE + SUGGESTIONS
    # ------------------------------------------------------
    def predict(self, text: str) -> Dict:
        """
        Predict with SVM.
        If confidence is low, return fallback taxonomy suggestions.
        """

        # First ensure classifier exists
        if self.pipeline is None:
            raise RuntimeError("Pipeline not trained. Call fit_classifier() first.")

        # 1) Classifier prediction
        probs = self.pipeline.predict_proba([text])[0]
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_sysid = self.label_encoder.inverse_transform([top_idx])[0]

        result = {
            "svm_sysid": top_sysid,
            "svm_confidence": top_prob,
            "is_confident": top_prob >= self.conf_threshold
        }

        # 2) If confident, return early
        if result["is_confident"] or self.tax_mat is None:
            return result

        # 3) Else fallback to taxonomy suggestions
        qv = self.retr_vec.transform([text])
        sims = cosine_similarity(qv, self.tax_mat).ravel()
        top_k_idx = np.argsort(-sims)[:5]

        suggestions = [
            {
                "sysid": self.tax_sysids[i],
                "label": self.tax_labels[i],
                "similarity": float(sims[i])
            }
            for i in top_k_idx
        ]

        result["suggestions"] = suggestions
        return result


    # ------------------------------------------------------
    # 5. SEARCH (taxonomy embedding)
    # ------------------------------------------------------
    def search(self, text: str, topk: int = 5) -> List[Dict]:
        if self.retr_vec is None or self.tax_mat is None:
            raise RuntimeError("Retrieval not built. Call fit_taxonomy().")

        qv = self.retr_vec.transform([text])
        sims = cosine_similarity(qv, self.tax_mat).ravel()
        idx = np.argsort(-sims)[:topk]

        return [
            {
                "sysid": self.tax_sysids[i],
                "label": self.tax_labels[i],
                "similarity": float(sims[i])
            }
            for i in idx
        ]


    # ------------------------------------------------------
    # 6. SAVE & LOAD
    # ------------------------------------------------------
    def save(self, path: str):
        obj = {
            "version": 1,
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "retr_vec": self.retr_vec,
            "tax_mat": self.tax_mat,
            "tax_sysids": self.tax_sysids,
            "tax_labels": self.tax_labels,
            "tfidf_kwargs": self.tfidf_kwargs,
            "conf_threshold": self.conf_threshold
        }
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        inst = cls(
            confidence_threshold=obj.get("conf_threshold", 0.45),
            **obj.get("tfidf_kwargs", {})
        )
        inst.pipeline = obj["pipeline"]
        inst.label_encoder = obj["label_encoder"]
        inst.retr_vec = obj["retr_vec"]
        inst.tax_mat = obj["tax_mat"]
        inst.tax_sysids = obj["tax_sysids"]
        inst.tax_labels = obj["tax_labels"]
        return inst