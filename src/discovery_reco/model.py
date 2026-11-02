"""Sparse content-profile recommender with portable artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

EVENT_WEIGHT = {"impression": 0.0, "click": 0.25, "play": 0.5, "complete": 1.0, "like": 1.5, "dismiss": -1.0}


@dataclass
class Recommendation:
    item_id: str
    title: str
    content_type: str
    score: float
    reason: str


class DiscoveryModel:
    def __init__(
        self,
        catalog: pd.DataFrame,
        matrix: sparse.csr_matrix,
        users: dict[str, np.ndarray],
        seen: dict[str, set[str]] | None = None,
    ):
        self.catalog = catalog.reset_index(drop=True)
        self.matrix = normalize(matrix.tocsr(), copy=False)
        self.users = users
        self.seen = seen or {}
        self.item_index = {x: i for i, x in enumerate(self.catalog.item_id)}

    def recommend(self, user_id: str, limit: int = 10, content_type: str | None = None) -> tuple[str, list[Recommendation]]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        eligible = np.ones(len(self.catalog), dtype=bool)
        if content_type:
            eligible &= self.catalog.content_type.to_numpy() == content_type
        profile = self.users.get(str(user_id))
        if profile is None:
            strategy = "cold_start_discovery"
            scores = self.catalog.discovery_score.to_numpy(float)
        else:
            strategy = "behavior_content_profile"
            affinity = np.asarray(self.matrix @ profile).ravel()
            scores = 0.85 * affinity + 0.15 * self.catalog.discovery_score.to_numpy(float)
            for item_id in self.seen.get(str(user_id), set()):
                if item_id in self.item_index:
                    eligible[self.item_index[item_id]] = False
        scores = scores.copy()
        scores[~eligible] = -np.inf
        count = min(limit, int(np.isfinite(scores).sum()))
        if not count:
            return strategy, []
        chosen = np.argpartition(scores, -count)[-count:]
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        output = []
        for idx in chosen:
            row = self.catalog.iloc[idx]
            reason = "Popular, well-supported discovery pick"
            if profile is not None:
                reason = f"Matches your {', '.join(row.genres_list[:2]) or row.content_type} history"
            output.append(Recommendation(row.item_id, row.title, row.content_type, float(scores[idx]), reason))
        return strategy, output

    def save(self, destination: str | Path) -> Path:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        self.catalog.to_json(root / "catalog.jsonl", orient="records", lines=True, force_ascii=False)
        sparse.save_npz(root / "features.npz", self.matrix)
        np.savez_compressed(root / "users.npz", **self.users)
        (root / "seen.json").write_text(
            json.dumps({user: sorted(items) for user, items in self.seen.items()}, sort_keys=True) + "\n"
        )
        (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "items": len(self.catalog), "users": len(self.users)}, indent=2) + "\n")
        return root

    @classmethod
    def load(cls, source: str | Path) -> "DiscoveryModel":
        root = Path(source)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported artifact schema")
        catalog = pd.read_json(root / "catalog.jsonl", lines=True)
        features = sparse.load_npz(root / "features.npz")
        with np.load(root / "users.npz", allow_pickle=False) as values:
            users = {name: values[name] for name in values.files}
        seen = {user: set(items) for user, items in json.loads((root / "seen.json").read_text()).items()}
        if features.shape[0] != len(catalog) or len(catalog) != manifest["items"]:
            raise ValueError("artifact shape mismatch")
        return cls(catalog, features, users, seen)


def train(catalog: pd.DataFrame, events: pd.DataFrame) -> DiscoveryModel:
    vectorizer = TfidfVectorizer(min_df=1, max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(catalog.feature_text).tocsr()
    item_index = {x: i for i, x in enumerate(catalog.item_id)}
    users: dict[str, np.ndarray] = {}
    seen: dict[str, set[str]] = {}
    if not events.empty:
        latest = events.timestamp.max()
        for user_id, group in events.groupby("user_id"):
            seen[str(user_id)] = set(group.loc[group.event_type != "impression", "item_id"].astype(str))
            rows, weights = [], []
            for event in group.itertuples():
                weight = EVENT_WEIGHT.get(event.event_type, 0.0)
                age_days = max(0.0, (latest - event.timestamp).total_seconds() / 86400)
                weight *= 0.5 ** (age_days / 180)
                if weight and event.item_id in item_index:
                    rows.append(item_index[event.item_id]); weights.append(weight)
            if rows:
                profile = np.asarray(matrix[rows].T @ np.asarray(weights)).ravel()
                norm = np.linalg.norm(profile)
                if norm:
                    users[str(user_id)] = profile / norm
    return DiscoveryModel(catalog.drop(columns=["feature_text"]), matrix, users, seen)
