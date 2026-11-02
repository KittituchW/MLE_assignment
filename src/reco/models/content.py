"""Metadata model for cold-start catalogue reach."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from reco.models.base import Recommender


def fit_content_embeddings(catalog: pd.DataFrame, components: int = 128, seed: int = 42) -> np.ndarray:
    """Fit TF-IDF plus truncated SVD content embeddings."""
    documents = (
        catalog["title"].fillna("").astype(str)
        + " "
        + catalog["genres"].map(lambda genres: " ".join(genres) + " " + " ".join(genres))
        + " "
        + catalog["description"].fillna("").astype(str)
    )
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        min_df=2,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(documents)
    n_components = min(components, max(2, min(features.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embeddings = svd.fit_transform(features).astype(np.float32)
    return normalize(embeddings, axis=1, norm="l2").astype(np.float32)


class ContentRecommender(Recommender):
    """Score by similarity to the watch-weighted user content vector."""

    name = "content"

    def __init__(self, embeddings: np.ndarray, history: sparse.csr_matrix) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.history = history

    def score(self, user_index: int) -> np.ndarray:
        """Return content cosine-like scores for one user."""
        row = self.history.getrow(user_index)
        if row.nnz == 0:
            return np.zeros(self.embeddings.shape[0], dtype=np.float32)
        user_vector = np.asarray(row @ self.embeddings).ravel()
        norm = float(np.linalg.norm(user_vector))
        if norm <= 0:
            return np.zeros(self.embeddings.shape[0], dtype=np.float32)
        return (self.embeddings @ (user_vector / norm)).astype(np.float32)
