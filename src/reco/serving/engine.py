"""In-memory scoring and explanation engine."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from reco.bundle.reader import BundleData
from reco.serving.ranking import rank_items

Strategy = Literal["personalized", "content_based", "cold_start_popular", "degraded_popular"]


def _minmax(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    low, high = float(np.min(result)), float(np.max(result))
    if high <= low:
        return np.zeros_like(result)
    return ((result - low) / (high - low)).astype(np.float32)


class RecommendationEngine:
    """Serve hybrid recommendations from one immutable bundle."""

    def __init__(self, bundle: BundleData) -> None:
        self.bundle = bundle
        self.user_index = {user_id: index for index, user_id in enumerate(bundle.user_ids)}
        serving = bundle.manifest.get("serving", {})
        self.weights = tuple(float(value) for value in serving.get("hybrid_weights", (0.6, 0.3, 0.1)))
        if len(self.weights) != 3 or not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Bundle hybrid weights must contain three values summing to 1")
        self.diversity_fraction = float(serving.get("diversity_fraction", 0.4))

    def _popularity(self) -> np.ndarray:
        return _minmax(self.bundle.popularity)

    def _content(self, user_index: int) -> np.ndarray:
        history = self.bundle.history.getrow(user_index)
        if history.nnz == 0:
            return np.zeros(len(self.bundle.item_ids), dtype=np.float32)
        user_vector = np.asarray(history @ self.bundle.embeddings).ravel()
        norm = float(np.linalg.norm(user_vector))
        if norm <= 0:
            return np.zeros(len(self.bundle.item_ids), dtype=np.float32)
        return (self.bundle.embeddings @ (user_vector / norm)).astype(np.float32)

    def _personalized(self, user_index: int) -> np.ndarray:
        history = self.bundle.history.getrow(user_index)
        item_scores = (history @ self.bundle.similarity).toarray().ravel().astype(np.float32)
        item_weight, content_weight, popularity_weight = self.weights
        return (
            item_weight * _minmax(item_scores)
            + content_weight * _minmax(self._content(user_index))
            + popularity_weight * self._popularity()
        ).astype(np.float32)

    def _reason(self, user_index: int | None, strategy: str) -> str:
        """Do not disclose a user's watched titles through this unauthenticated demo API."""
        if user_index is None or strategy in {"cold_start_popular", "degraded_popular"}:
            return "Popular with viewers"
        return "Based on your viewing preferences"

    def recommend(
        self, user_id: str, limit: int = 10, content_type: str | None = None
    ) -> tuple[Strategy, list[dict[str, Any]]]:
        """Score, apply ranking rules, and return strategy plus response item dictionaries."""
        user_index = self.user_index.get(user_id)
        watched: set[int] = set()
        strategy: Strategy
        try:
            if user_index is None:
                strategy = "cold_start_popular"
                scores = self._popularity()
            else:
                history = self.bundle.history.getrow(user_index)
                watched = set(int(index) for index in history.indices)
                if history.nnz < 3:
                    strategy = "content_based"
                    scores = self._content(user_index)
                else:
                    strategy = "personalized"
                    scores = self._personalized(user_index)
        except Exception:
            strategy = "degraded_popular"
            scores = self._popularity()
        ranked = rank_items(
            scores,
            self.bundle.catalog,
            watched,
            limit,
            content_type,
            diversity_fraction=self.diversity_fraction,
        )
        normalized = _minmax(scores)
        reason = self._reason(user_index, strategy)
        items: list[dict[str, Any]] = []
        for rank, item_index in enumerate(ranked, start=1):
            item = self.bundle.catalog[item_index]
            items.append(
                {
                    "rank": rank,
                    "item_id": item["item_id"],
                    "title": item["title"],
                    "content_type": item["content_type"],
                    "genres": item["genres"],
                    "release_year": item["release_year"],
                    "score": round(float(normalized[item_index]), 6),
                    "reason": reason,
                }
            )
        return strategy, items
