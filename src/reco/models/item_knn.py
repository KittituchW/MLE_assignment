"""Sparse item-item collaborative filtering."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from reco.models.base import Recommender


def prune_top_k(matrix: sparse.spmatrix, k: int) -> sparse.csr_matrix:
    """Keep at most the largest ``k`` non-zero values in each sparse row."""
    if k < 1:
        raise ValueError("k must be positive")
    source = matrix.tocsr().astype(np.float32)
    source.setdiag(0)
    source.eliminate_zeros()
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    indptr = [0]
    for row in range(source.shape[0]):
        start, end = source.indptr[row], source.indptr[row + 1]
        row_data = source.data[start:end]
        row_indices = source.indices[start:end]
        if len(row_data) > k:
            chosen = np.argpartition(row_data, -k)[-k:]
            order = chosen[np.argsort(row_data[chosen])[::-1]]
            row_data = row_data[order]
            row_indices = row_indices[order]
        data.append(row_data)
        indices.append(row_indices)
        indptr.append(indptr[-1] + len(row_data))
    return sparse.csr_matrix(
        (
            np.concatenate(data) if data else np.array([], dtype=np.float32),
            np.concatenate(indices) if indices else np.array([], dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=source.shape,
        dtype=np.float32,
    )


def fit_item_similarity(history: sparse.csr_matrix, neighbors: int = 100) -> sparse.csr_matrix:
    """Fit cosine item similarity without constructing a dense all-pairs matrix."""
    normalized = normalize(history, axis=0, norm="l2", copy=True)
    similarity = normalized.T @ normalized
    return prune_top_k(similarity, neighbors)


class ItemKNNRecommender(Recommender):
    """Score from a user's weighted history and item neighbours."""

    name = "item_knn"

    def __init__(self, similarity: sparse.csr_matrix, history: sparse.csr_matrix) -> None:
        self.similarity = similarity
        self.history = history

    def score(self, user_index: int) -> np.ndarray:
        """Return sparse history-times-similarity scores as a dense vector."""
        return (self.history.getrow(user_index) @ self.similarity).toarray().ravel().astype(np.float32)
