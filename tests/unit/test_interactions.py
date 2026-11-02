from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from reco.interactions.simulator import SimulatedInteractionSource
from reco.interactions.source import ParquetInteractionSource


def _cfg(seed: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        as_of_date="2026-01-31",
        simulator=SimpleNamespace(
            users=6,
            event_window_days=180,
            fixed_demo_events=8,
            min_events=5,
            max_events=10,
            lognormal_mean=2,
            lognormal_sigma=0.2,
        ),
    )


def _catalog() -> pd.DataFrame:
    genres = [["Action"], ["Romance"], ["Documentary"], ["Comedy"], ["Action", "Adventure"]] * 4
    return pd.DataFrame(
        {
            "item_id": [f"movie_{index}" for index in range(20)],
            "genres": genres,
            "language": ["en", "ko", "en", "ja", "en"] * 4,
            "popularity_percentile": [(index + 1) / 20 for index in range(20)],
            "release_date": ["2025-01-01"] * 19 + ["2026-02-01"],
            "recommendable": [True] * 20,
        }
    )


def test_simulator_is_deterministic_and_respects_release_date() -> None:
    first = SimulatedInteractionSource(_catalog(), _cfg()).load()
    second = SimulatedInteractionSource(_catalog(), _cfg()).load()
    pd.testing.assert_frame_equal(first, second)
    assert "movie_19" not in set(first.item_id)
    releases = _catalog().set_index("item_id").release_date
    assert all(row.event_ts >= pd.Timestamp(releases[row.item_id], tz="UTC") for row in first.itertuples())


def test_parquet_source_round_trip(tmp_path: Path) -> None:
    expected = SimulatedInteractionSource(_catalog(), _cfg()).load()
    path = tmp_path / "events.parquet"
    expected.to_parquet(path, index=False)
    actual = ParquetInteractionSource(path).load()
    pd.testing.assert_frame_equal(expected, actual)
