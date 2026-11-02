"""Fail-fast raw catalogue validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from loguru import logger

from reco.config import runtime_paths

REQUIRED_COLUMNS = {
    "show_id",
    "type",
    "title",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "genres",
    "language",
    "description",
    "popularity",
    "vote_count",
    "vote_average",
}


def _schema(content_type: str) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "show_id": pa.Column(int, nullable=False, coerce=True),
        "type": pa.Column(str, nullable=False, coerce=True),
        "title": pa.Column(str, nullable=False, coerce=True),
        "release_year": pa.Column(int, pa.Check.in_range(1900, 2026), nullable=False, coerce=True),
        "popularity": pa.Column(float, pa.Check.ge(0), nullable=False, coerce=True),
        "vote_count": pa.Column(int, pa.Check.ge(0), nullable=False, coerce=True),
        "vote_average": pa.Column(float, pa.Check.in_range(0, 10), nullable=False, coerce=True),
    }
    return pa.DataFrameSchema(columns, strict=False, coerce=True, name=f"raw_{content_type}")


def validate(cfg: Any, home: Path | None = None) -> dict[str, int]:
    """Validate raw schemas and return observed row counts."""
    raw = runtime_paths(home)["raw"]
    counts: dict[str, int] = {}
    for key in ("movies", "tv"):
        path = raw / str(cfg.data[key].filename)
        if not path.is_file():
            raise FileNotFoundError(f"Raw file missing; run ingest first: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        try:
            _schema(key).validate(frame, lazy=True)
        except pa.errors.SchemaErrors as exc:
            raise ValueError(f"Data-quality gate failed for {path.name}:\n{exc.failure_cases.head(20)}") from exc
        if frame.empty:
            raise ValueError(f"Data-quality gate failed: {path.name} is empty")
        counts[key] = len(frame)
        logger.info("validated {} rows from {}", len(frame), path.name)
    return counts
