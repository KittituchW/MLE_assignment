"""Dependency-light ranking metrics with explicit semantics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Return hits divided by the count of relevant items."""
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Return binary normalized discounted cumulative gain."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def catalogue_coverage(recommendations: Iterable[Sequence[str]], catalogue_size: int) -> float:
    """Return unique recommended items divided by recommendable catalogue size."""
    if catalogue_size <= 0:
        return 0.0
    return len({item for recommendation in recommendations for item in recommendation}) / catalogue_size
