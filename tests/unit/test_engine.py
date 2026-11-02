from pathlib import Path

import numpy as np
from scipy import sparse

from reco.bundle.reader import BundleData
from reco.serving.engine import RecommendationEngine


def _engine(history: sparse.csr_matrix) -> RecommendationEngine:
    count = 6
    catalog = [
        {
            "item_id": f"movie_{index}",
            "title": f"Title {index}",
            "normalized_title": f"title {index}",
            "content_type": "movie",
            "genres": ["Action", f"G{index}"],
            "release_year": 2020,
            "recommendable": True,
        }
        for index in range(count)
    ]
    bundle = BundleData(
        Path("bundle"),
        {"version": "v1"},
        catalog,
        ["A"],
        [item["item_id"] for item in catalog],
        sparse.csr_matrix(np.ones((count, count), dtype=np.float32) - np.eye(count, dtype=np.float32)),
        history,
        np.eye(count, dtype=np.float32),
        np.linspace(0, 1, count, dtype=np.float32),
    )
    return RecommendationEngine(bundle)


def test_content_based_branch_for_short_history() -> None:
    engine = _engine(sparse.csr_matrix([[1, 0, 0, 0, 0, 0]], dtype=np.float32))
    strategy, items = engine.recommend("A", limit=3)
    assert strategy == "content_based"
    assert items
    assert all(item["item_id"] != "movie_0" for item in items)


def test_scoring_exception_degrades_to_popularity(monkeypatch) -> None:
    engine = _engine(sparse.csr_matrix([[1, 1, 1, 0, 0, 0]], dtype=np.float32))

    def fail(_: int) -> np.ndarray:
        raise RuntimeError("blas failure")

    monkeypatch.setattr(engine, "_personalized", fail)
    strategy, items = engine.recommend("A", limit=2)
    assert strategy == "degraded_popular"
    assert items
