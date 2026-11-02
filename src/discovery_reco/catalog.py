"""Catalog preparation based on findings from the supplied EDA."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _text(value: object) -> str:
    return "" if pd.isna(value) else " ".join(str(value).split())


def _tokens(value: object) -> list[str]:
    text = _text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return sorted({_text(x) for x in parsed if _text(x)})
        except (SyntaxError, ValueError):
            pass
    return sorted({part.strip() for part in text.replace("|", ",").split(",") if part.strip()})


def _find_csv(source: Path, kind: str) -> Path:
    matches = sorted(source.glob(f"*{kind}*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one {kind} CSV in {source}, found {len(matches)}")
    return matches[0]


def prepare_catalog(source: str | Path) -> pd.DataFrame:
    """Create one globally unique, model-ready catalog.

    EDA-driven rules: namespace cross-type IDs, deterministically collapse duplicate TV IDs,
    treat zero ratings as missing, ignore misleading date_added/duration, and calibrate
    popularity and ratings within content type.
    """
    root = Path(source)
    frames = []
    for pattern, kind in (("movies", "movie"), ("tv_shows", "tv")):
        frame = pd.read_csv(_find_csv(root, pattern))
        frame["content_type"] = kind
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("show_id", "release_year", "vote_average", "vote_count", "popularity"):
        data[col] = pd.to_numeric(data.get(col), errors="coerce")
    data["item_id"] = data["content_type"] + "_" + data["show_id"].astype("Int64").astype(str)
    data = (data.sort_values(["item_id", "vote_count", "popularity"], ascending=[True, False, False])
            .drop_duplicates("item_id", keep="first").reset_index(drop=True))
    data["title"] = data["title"].map(_text)
    data["description"] = data.get("description", "").map(_text)
    data["language"] = data.get("language", data.get("original_language", "")).map(_text)
    data["genres_list"] = data.get("genres", "").map(_tokens)
    data["countries_list"] = data.get("country", "").map(_tokens)
    data["vote_average_clean"] = data["vote_average"].where(data["vote_average"] > 0)
    data["vote_count"] = data["vote_count"].fillna(0).clip(lower=0)
    rated = data[data["vote_average_clean"].notna() & (data["vote_count"] > 0)]
    priors = rated.groupby("content_type")["vote_average_clean"].median().to_dict()
    strength = rated.groupby("content_type")["vote_count"].quantile(0.60).clip(lower=10).to_dict()
    data["quality_score"] = 0.0
    for kind, idx in data.groupby("content_type").groups.items():
        votes = data.loc[idx, "vote_count"].to_numpy(float)
        observed = data.loc[idx, "vote_average_clean"].fillna(priors[kind]).to_numpy(float)
        m = float(strength[kind])
        data.loc[idx, "quality_score"] = (votes * observed + m * priors[kind]) / (votes + m)
    data["popularity_pct"] = data.groupby("content_type")["popularity"].rank(pct=True).fillna(0)
    year = data["release_year"].fillna(data["release_year"].median()).clip(2010, 2025)
    data["recency_score"] = (year - 2010) / 15
    data["discovery_score"] = (
        0.50 * data["popularity_pct"]
        + 0.35 * (data["quality_score"] / 10)
        + 0.15 * data["recency_score"]
    )
    data["feature_text"] = [
        " ".join(
            [f"genre_{g.replace(' ', '_')}" for g in genres] * 3
            + [f"lang_{language}"] * 2
            + [f"country_{c.replace(' ', '_')}" for c in countries]
            + [kind, title, description]
        )
        for genres, language, countries, kind, title, description in zip(
            data["genres_list"], data["language"], data["countries_list"],
            data["content_type"], data["title"], data["description"], strict=True
        )
    ]
    return data[["item_id", "title", "content_type", "release_year", "genres_list", "language",
                 "vote_count", "quality_score", "discovery_score", "feature_text"]]


def write_catalog(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def known_items(events: pd.DataFrame, catalog_ids: Iterable[str]) -> pd.DataFrame:
    """Validate and retain behavior that can be joined to the current catalog."""
    required = {"user_id", "item_id", "event_type", "timestamp"}
    missing = required - set(events)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    result = events.copy()
    result["user_id"] = result["user_id"].astype(str)
    result["item_id"] = result["item_id"].astype(str)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result[result["item_id"].isin(set(catalog_ids)) & result["timestamp"].notna()]
    return result.sort_values("timestamp").reset_index(drop=True)

