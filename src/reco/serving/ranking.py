"""Post-model business rules in their documented order."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def rank_items(
    scores: np.ndarray,
    catalog: Sequence[dict],
    watched: set[int],
    limit: int,
    content_type: str | None = None,
    *,
    diversity_fraction: float = 0.4,
) -> list[int]:
    """Apply eligibility, history, filter, duplicate, and diversity rules before cutting to ``limit``."""
    values = np.asarray(scores, dtype=np.float32)
    eligible = np.fromiter((bool(item["recommendable"]) for item in catalog), dtype=bool, count=len(catalog))
    for item_index in watched:
        if 0 <= item_index < len(eligible):
            eligible[item_index] = False
    if content_type:
        eligible &= np.fromiter(
            (str(item["content_type"]) == content_type for item in catalog), dtype=bool, count=len(catalog)
        )
    candidates = np.flatnonzero(eligible & np.isfinite(values))
    candidates = candidates[np.argsort(values[candidates], kind="stable")[::-1]]

    result: list[int] = []
    seen_titles: set[tuple[str, int]] = set()
    genre_counts: Counter[tuple[str, ...]] = Counter()
    if not 0 < diversity_fraction <= 1:
        raise ValueError("diversity_fraction must be in (0, 1]")
    genre_cap = max(1, int(np.floor(limit * diversity_fraction)))
    for candidate_index in candidates:
        item_index = int(candidate_index)
        item = catalog[item_index]
        title_key = (str(item["normalized_title"]), int(item["release_year"]))
        if title_key in seen_titles:
            continue
        genre_key = tuple(sorted(str(genre) for genre in item["genres"]))
        if genre_counts[genre_key] >= genre_cap:
            continue
        seen_titles.add(title_key)
        genre_counts[genre_key] += 1
        result.append(item_index)
        if len(result) == limit:
            break
    return result
