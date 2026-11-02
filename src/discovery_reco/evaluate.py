"""Leakage-resistant temporal evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import train


def temporal_evaluate(catalog: pd.DataFrame, events: pd.DataFrame, k: int = 10) -> dict[str, float]:
    positives = events[events.event_type.isin(["play", "complete", "like"])].copy()
    sizes = positives.groupby("user_id").size()
    users = set(sizes[sizes >= 2].index)
    positives = positives[positives.user_id.isin(users)].sort_values("timestamp")
    test = positives.groupby("user_id").tail(1)
    train_events = events.drop(index=test.index)
    model = train(catalog, train_events)
    recalls, reciprocal, all_items = [], [], set()
    for row in test.itertuples():
        _, recs = model.recommend(str(row.user_id), limit=k)
        ids = [x.item_id for x in recs]
        all_items.update(ids)
        recalls.append(float(row.item_id in ids))
        reciprocal.append(1 / (ids.index(row.item_id) + 1) if row.item_id in ids else 0.0)
    return {"users": len(test), f"hit_rate@{k}": float(np.mean(recalls)) if recalls else 0.0,
            f"mrr@{k}": float(np.mean(reciprocal)) if reciprocal else 0.0,
            "catalog_coverage": len(all_items) / len(catalog) if len(catalog) else 0.0}

