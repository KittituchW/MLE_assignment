"""In-memory scoring and explanation engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from reco.bundle.reader import BundleData
from reco.serving.ranking import rank_items


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
        return 0.6 * _minmax(item_scores) + 0.3 * _minmax(self._content(user_index)) + 0.1 * self._popularity()

    def _reason(self, user_index: int | None, strategy: str) -> str:
        if user_index is None or strategy in {"cold_start_popular", "degraded_popular"}:
            return "Popular with viewers"
        history = self.bundle.history.getrow(user_index)
        if history.nnz == 0:
            return "Similar to content in your profile"
        watched_index = int(history.indices[int(np.argmax(history.data))])
        return f"Because you watched {self.bundle.catalog[watched_index]['title']}"

    def recommend(
        self, user_id: str, limit: int = 10, content_type: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Score, apply ranking rules, and return strategy plus response item dictionaries."""
        user_index = self.user_index.get(user_id)
        watched: set[int] = set()
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
        ranked = rank_items(scores, self.bundle.catalog, watched, limit, content_type)
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
