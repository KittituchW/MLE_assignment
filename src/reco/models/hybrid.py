"""Blended recommender used by serving."""

from __future__ import annotations

import numpy as np

from reco.models.base import Recommender, minmax


class HybridRecommender(Recommender):
    """Blend independently normalized collaborative, content, and popularity scores."""

    name = "hybrid"

    def __init__(
        self,
        item_knn: Recommender,
        content: Recommender,
        popularity: Recommender,
        weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    ) -> None:
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Hybrid weights must sum to one")
        self.item_knn = item_knn
        self.content = content
        self.popularity = popularity
        self.weights = weights

    def score(self, user_index: int) -> np.ndarray:
        """Return the configured score blend."""
        item_weight, content_weight, popularity_weight = self.weights
        return (
            item_weight * minmax(self.item_knn.score(user_index))
            + content_weight * minmax(self.content.score(user_index))
            + popularity_weight * minmax(self.popularity.score(user_index))
        ).astype(np.float32)
