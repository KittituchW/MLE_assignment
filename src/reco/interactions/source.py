"""Behavior source interface and real-event Parquet adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

INTERACTION_COLUMNS = ["user_id", "item_id", "event_ts", "event_type", "watch_ratio"]


class InteractionSource(Protocol):
    """Interface implemented by simulated and real event sources."""

    def load(self) -> pd.DataFrame:
        """Return canonical interactions."""
        ...


def validate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize source events."""
    missing = set(INTERACTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Interactions missing columns: {sorted(missing)}")
    result = frame[INTERACTION_COLUMNS].copy()
    result["event_ts"] = pd.to_datetime(result["event_ts"], utc=True)
    result["watch_ratio"] = pd.to_numeric(result["watch_ratio"], errors="raise").astype(float)
    if not result["watch_ratio"].between(0, 1).all():
        raise ValueError("watch_ratio must be between 0 and 1")
    if result[["user_id", "item_id", "event_type"]].isna().any().any():
        raise ValueError("Interaction identifiers and event_type must not be null")
    return result.sort_values(["user_id", "event_ts", "item_id"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class ParquetInteractionSource:
    """Read production-shaped behavior events from Parquet."""

    path: Path

    def load(self) -> pd.DataFrame:
        """Read and validate the configured file."""
        return validate_interactions(pd.read_parquet(self.path))
