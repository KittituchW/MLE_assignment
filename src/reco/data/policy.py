"""Layered catalogue policy used before recommendation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reco.config import project_root


def load_denylist(path: Path) -> set[str]:
    """Load item IDs while allowing comments and blank lines."""
    return {
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def apply_content_policy(
    frame: pd.DataFrame,
    policy_path: Path | None = None,
    denylist_path: Path | None = None,
) -> pd.DataFrame:
    """Add ``policy_blocked`` and ``policy_reason`` using the three specified layers."""
    policy_file = policy_path or project_root() / "configs" / "content_policy.yaml"
    deny_file = denylist_path or project_root() / "configs" / "denylist_items.txt"
    policy: dict[str, Any] = yaml.safe_load(policy_file.read_text(encoding="utf-8"))
    keywords = "|".join(f"(?:{keyword})" for keyword in policy["keywords"])
    pattern = re.compile(rf"\b(?:{keywords})\b", re.IGNORECASE)
    text = frame["title"].fillna("").astype(str) + " " + frame["description"].fillna("").astype(str)
    keyword_match = text.str.contains(pattern, regex=True, na=False)

    genres = frame["genres"].map(set)
    evidence = policy["evidence_floor"]
    evidence_match = (
        frame["content_type"].eq(evidence["content_type"])
        & genres.map(lambda values: bool(set(evidence["genres"]) & values))
        & frame["vote_count"].le(int(evidence["max_vote_count"]))
        & frame["budget"].fillna(0).eq(float(evidence["budget"]))
    )
    deny_match = frame["item_id"].isin(load_denylist(deny_file))

    result = frame.copy()
    result["policy_blocked"] = keyword_match | evidence_match | deny_match
    result["policy_reason"] = ""
    result.loc[keyword_match, "policy_reason"] = "keyword"
    result.loc[evidence_match & ~keyword_match, "policy_reason"] = "evidence_floor"
    result.loc[deny_match & ~keyword_match & ~evidence_match, "policy_reason"] = "denylist"
    return result
