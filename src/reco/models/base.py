"""Common recommender contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Recommender(ABC):
    """A model that scores every catalogue item for one indexed user."""

    name: str

    @abstractmethod
    def score(self, user_index: int) -> np.ndarray:
        """Return one float32 score per item."""


def minmax(values: np.ndarray) -> np.ndarray:
    """Normalize finite values to [0, 1], returning zeros for a constant vector."""
    result = np.asarray(values, dtype=np.float32)
    low = float(np.nanmin(result))
    high = float(np.nanmax(result))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(result, dtype=np.float32)
    return ((result - low) / (high - low)).astype(np.float32, copy=False)
