"""Leakage-safe temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Train, validation, and final test event frames with explicit boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp


def temporal_split(
    interactions: pd.DataFrame, as_of_date: str, window_days: int = 14, validation_days: int | None = None
) -> TemporalSplit:
    """Reserve two final windows, and remove already-seen targets per user."""
    events = interactions.copy()
    events["event_ts"] = pd.to_datetime(events.event_ts, utc=True)
    as_of = pd.Timestamp(as_of_date, tz="UTC")
    test_start = as_of - pd.Timedelta(days=window_days)
    validation_start = test_start - pd.Timedelta(days=validation_days if validation_days is not None else window_days)
    train = events.loc[events.event_ts < validation_start].copy()
    validation = events.loc[(events.event_ts >= validation_start) & (events.event_ts < test_start)].copy()
    test = events.loc[(events.event_ts >= test_start) & (events.event_ts <= as_of)].copy()

    train_pairs = set(zip(train.user_id.astype(str), train.item_id.astype(str), strict=True))
    validation = validation.loc[
        [
            (str(user), str(item)) not in train_pairs
            for user, item in zip(validation.user_id, validation.item_id, strict=True)
        ]
    ]
    pretest_pairs = train_pairs | set(zip(validation.user_id.astype(str), validation.item_id.astype(str), strict=True))
    test = test.loc[
        [(str(user), str(item)) not in pretest_pairs for user, item in zip(test.user_id, test.item_id, strict=True)]
    ]
    return TemporalSplit(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
        validation_start,
        test_start,
    )
