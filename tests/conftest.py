from pathlib import Path

import pandas as pd
import pytest

from discovery_reco.catalog import prepare_catalog


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return Path(__file__).parents[1] / "data" / "sample"


@pytest.fixture(scope="session")
def catalog(sample_dir: Path) -> pd.DataFrame:
    return prepare_catalog(sample_dir)


@pytest.fixture()
def events(catalog: pd.DataFrame) -> pd.DataFrame:
    action = catalog[catalog.genres_list.map(lambda x: "Action" in x)].head(4).item_id.tolist()
    drama = catalog[catalog.genres_list.map(lambda x: "Drama" in x)].head(4).item_id.tolist()
    rows = []
    for user, items in (("A", action), ("B", drama)):
        for day, item in enumerate(items):
            rows.append((user, item, "complete", f"2026-01-{day + 1:02d}T00:00:00Z"))
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "event_type", "timestamp"])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame

