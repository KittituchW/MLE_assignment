"""Popularity and quality cold-start baseline."""

from __future__ import annotations

import numpy as np

from reco.models.base import Recommender


class PopularityRecommender(Recommender):
    """Return the same catalogue-quality score for every user."""

    name = "popularity"

    def __init__(self, popularity: np.ndarray) -> None:
        self.popularity = np.asarray(popularity, dtype=np.float32)

    def score(self, user_index: int) -> np.ndarray:
        """Return catalogue-quality scores; ``user_index`` is intentionally unused."""
        del user_index
        return self.popularity.copy()
