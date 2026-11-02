"""Deterministic synthetic behavior generator for a catalogue without users."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reco.config import project_root
from reco.interactions.source import validate_interactions


@dataclass
class SimulatedInteractionSource:
    """Generate behavior whose assumptions are transparent and reproducible."""

    catalog: pd.DataFrame
    cfg: Any
    personas_path: Path | None = None

    def _personas(self) -> dict[str, Any]:
        path = self.personas_path or project_root() / "configs" / "personas.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def load(self) -> pd.DataFrame:
        """Generate deterministic interactions from the configured seed."""
        rng = np.random.default_rng(int(self.cfg.seed))
        personas = self._personas()
        persona_names = list(personas)
        demo = {"A": "action", "B": "romance_kdrama", "C": "documentary"}
        user_count = int(self.cfg.simulator.users)
        if user_count < 3:
            raise ValueError("simulator.users must be at least 3")
        users = list(demo) + [f"u{index:05d}" for index in range(user_count - 3)]
        assigned = [demo.get(user, persona_names[int(rng.integers(len(persona_names)))]) for user in users]

        items = self.catalog.loc[self.catalog.recommendable].reset_index(drop=True).copy()
        as_of = pd.Timestamp(str(self.cfg.as_of_date), tz="UTC")
        release = pd.to_datetime(items.release_date, utc=True)
        eligible = release.le(as_of).to_numpy()
        if not eligible.any():
            raise ValueError("No recommendable items were released by as_of_date")
        all_genres = sorted({genre for genres in items.genres for genre in genres})
        genre_index = {genre: index for index, genre in enumerate(all_genres)}
        genre_matrix = np.zeros((len(items), len(all_genres)), dtype=np.float32)
        for item_index, genres in enumerate(items.genres):
            for genre in genres:
                genre_matrix[item_index, genre_index[genre]] = 1.0
        norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
        genre_matrix = np.divide(genre_matrix, norms, out=np.zeros_like(genre_matrix), where=norms > 0)
        popularity = items.popularity_percentile.to_numpy(dtype=np.float64) ** 12
        languages = items.language.astype(str).to_numpy()

        rows: list[tuple[str, str, pd.Timestamp, str, float]] = []
        window_start = as_of - pd.Timedelta(days=int(self.cfg.simulator.event_window_days))
        for user, persona_name in zip(users, assigned, strict=True):
            persona = personas[persona_name]
            weights = np.full(len(all_genres), 0.05, dtype=np.float64)
            total_weight = float(sum(persona["genres"].values()))
            for genre, weight in persona["genres"].items():
                if genre in genre_index:
                    weights[genre_index[genre]] += 30.0 * float(weight) / total_weight
            preference = rng.dirichlet(weights)
            affinity = genre_matrix @ preference
            probability = (affinity + 0.003) ** 2 * popularity
            for language, boost in persona.get("language_boost", {}).items():
                probability[languages == str(language)] *= float(boost)
            probability[~eligible] = 0
            probability_sum = probability.sum()
            if probability_sum <= 0:
                probability = eligible.astype(np.float64)
                probability_sum = probability.sum()
            probability /= probability_sum
            event_count = (
                int(self.cfg.simulator.fixed_demo_events)
                if user in demo
                else int(
                    np.clip(
                        rng.lognormal(
                            float(self.cfg.simulator.lognormal_mean), float(self.cfg.simulator.lognormal_sigma)
                        ),
                        int(self.cfg.simulator.min_events),
                        int(self.cfg.simulator.max_events),
                    )
                )
            )
            event_count = min(event_count, int(np.count_nonzero(probability)))
            selected = rng.choice(len(items), size=event_count, replace=False, p=probability)
            maximum = max(float(affinity.max()), 1e-12)
            for item_index in selected:
                normalized_affinity = float(affinity[item_index] / maximum)
                ratio = float(rng.beta(1.0 + 5.0 * normalized_affinity, 2.0))
                earliest = max(window_start, release.iloc[item_index])
                span_seconds = max(0, int((as_of - earliest).total_seconds()))
                offset = int(rng.integers(span_seconds + 1)) if span_seconds else 0
                event_ts = earliest + pd.Timedelta(seconds=offset)
                event_type = "complete" if ratio >= 0.85 else "play"
                rows.append((user, str(items.item_id.iloc[item_index]), event_ts, event_type, ratio))
        frame = pd.DataFrame(rows, columns=["user_id", "item_id", "event_ts", "event_type", "watch_ratio"])
        return validate_interactions(frame)


def write_simulated(source: SimulatedInteractionSource, destination: Path) -> Path:
    """Write interactions deterministically and atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    source.load().to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)
    return destination
