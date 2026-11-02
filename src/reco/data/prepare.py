"""Catalogue cleaning and feature preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from reco.config import project_root, runtime_paths
from reco.data.policy import apply_content_policy

CATALOG_COLUMNS = [
    "item_id",
    "show_id",
    "content_type",
    "title",
    "normalized_title",
    "description",
    "genres",
    "language",
    "release_date",
    "release_year",
    "popularity",
    "popularity_percentile",
    "vote_count",
    "vote_average",
    "bayesian_rating",
    "budget",
    "revenue",
    "policy_blocked",
    "policy_reason",
    "recommendable",
]


def _genre_map(path: Path | None = None) -> dict[str, list[str]]:
    target = path or project_root() / "configs" / "genre_map.yaml"
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def harmonize_genres(value: Any, mapping: dict[str, list[str]]) -> list[str]:
    """Split, map, de-duplicate, and sort catalogue genres."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    output: list[str] = []
    for raw in str(value).split(","):
        genre = raw.strip()
        replacements = mapping.get(genre, [genre])
        output.extend(item for item in replacements if item)
    return sorted(set(output))


def namespace_item_id(content_type: str, show_id: int | str) -> str:
    """Make source IDs globally unique across movies and TV."""
    return f"{content_type}_{int(show_id)}"


def _normalise_title(value: str) -> str:
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


def _read_raw(cfg: Any, home: Path | None) -> pd.DataFrame:
    raw = runtime_paths(home)["raw"]
    frames: list[pd.DataFrame] = []
    for key, content_type in (("movies", "movie"), ("tv", "tv")):
        part = pd.read_csv(raw / str(cfg.data[key].filename))
        part["content_type"] = content_type
        if "budget" not in part:
            part["budget"] = np.nan
        if "revenue" not in part:
            part["revenue"] = np.nan
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare(cfg: Any, home: Path | None = None) -> Path:
    """Prepare the canonical catalogue and write ``catalog.parquet``."""
    paths = runtime_paths(home)
    frame = _read_raw(cfg, home)
    for numeric in ("show_id", "release_year", "popularity", "vote_count", "vote_average", "budget", "revenue"):
        frame[numeric] = pd.to_numeric(frame[numeric], errors="coerce")
    frame["item_id"] = [
        namespace_item_id(kind, show_id) for kind, show_id in zip(frame.content_type, frame.show_id, strict=True)
    ]
    frame = (
        frame.sort_values(["item_id", "popularity"], ascending=[True, False], kind="stable")
        .drop_duplicates("item_id", keep="first")
        .sort_values("item_id", kind="stable")
        .reset_index(drop=True)
    )
    mapping = _genre_map()
    frame["genres"] = frame["genres"].map(lambda value: harmonize_genres(value, mapping))
    for column in ("title", "description", "language"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["normalized_title"] = frame["title"].map(_normalise_title)
    frame["release_date"] = pd.to_datetime(frame["date_added"], errors="coerce").dt.strftime("%Y-%m-%d")
    missing_date = frame["release_date"].isna()
    frame.loc[missing_date, "release_date"] = (
        frame.loc[missing_date, "release_year"].astype("Int64").astype(str) + "-01-01"
    )
    frame["popularity_percentile"] = frame.groupby("content_type")["popularity"].rank(method="average", pct=True)

    rated = frame.loc[(frame.vote_count > 0) & (frame.vote_average > 0)]
    global_mean = float(np.average(rated.vote_average, weights=np.sqrt(rated.vote_count.clip(lower=1))))
    minimum_votes = float(rated.vote_count.quantile(0.60))
    votes = frame.vote_count.fillna(0).to_numpy(dtype=float)
    ratings = frame.vote_average.fillna(0).to_numpy(dtype=float)
    frame["bayesian_rating"] = (votes * ratings + minimum_votes * global_mean) / (votes + minimum_votes)

    frame = apply_content_policy(frame)
    metadata_usable = frame["genres"].map(bool) | frame["description"].str.strip().ne("")
    frame["recommendable"] = metadata_usable & ~frame["policy_blocked"]
    frame["budget"] = frame["budget"].astype("Float64")
    frame["revenue"] = frame["revenue"].astype("Float64")
    frame["show_id"] = frame["show_id"].astype(int)
    frame["release_year"] = frame["release_year"].astype(int)
    frame["vote_count"] = frame["vote_count"].astype(int)

    destination = paths["processed"] / "catalog.parquet"
    frame[CATALOG_COLUMNS].to_parquet(destination, index=False)
    summary = {
        "items": len(frame),
        "recommendable": int(frame.recommendable.sum()),
        "policy_blocked": int(frame.policy_blocked.sum()),
        "metadata_unusable": int((~metadata_usable).sum()),
    }
    (paths["processed"] / "catalog_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("prepared catalogue: {}", summary)
    return destination
