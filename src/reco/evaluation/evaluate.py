"""Full temporal evaluation of all configured recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reco.evaluation.metrics import catalogue_coverage, ndcg_at_k, recall_at_k
from reco.models.training import TrainedModels
from reco.serving.engine import RecommendationEngine
from reco.serving.ranking import rank_items


def evaluate_models(
    trained: TrainedModels,
    test: pd.DataFrame,
    max_users: int = 1500,
    k: int = 10,
    *,
    serving_engine: RecommendationEngine | None = None,
    diversity_fraction: float = 0.4,
) -> dict[str, dict[str, float]]:
    """Evaluate each model on new-to-user temporal targets."""
    item_index = {item: index for index, item in enumerate(trained.item_ids)}
    user_index = {user: index for index, user in enumerate(trained.user_ids)}
    grouped = {str(user): set(group.item_id.astype(str)) & set(item_index) for user, group in test.groupby("user_id")}
    users = [user for user in sorted(grouped) if user in user_index and grouped[user]][:max_users]
    eligible = trained.catalog.recommendable.to_numpy(dtype=bool)
    catalog = trained.catalog.to_dict("records")
    training_items = set(np.flatnonzero(np.asarray(trained.history.getnnz(axis=0)).ravel() > 0).tolist())
    results: dict[str, dict[str, float]] = {}
    for model_name, model in trained.models.items():
        recalls: list[float] = []
        ndcgs: list[float] = []
        recommendation_ids: list[list[str]] = []
        cold_count = 0
        total_count = 0
        for user in users:
            index = user_index[user]
            watched = trained.history.getrow(index).indices
            if model_name == "hybrid" and serving_engine is not None:
                _, served_items = serving_engine.recommend(user, limit=k)
                ranked = [item_index[item["item_id"]] for item in served_items]
            else:
                ranked = rank_items(
                    model.score(index),
                    catalog,
                    set(int(item) for item in watched),
                    k,
                    diversity_fraction=diversity_fraction,
                )
            recommended = [trained.item_ids[item] for item in ranked]
            relevant = grouped[user]
            recalls.append(recall_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))
            recommendation_ids.append(recommended)
            cold_count += sum(item not in training_items for item in ranked)
            total_count += len(ranked)
        results[model_name] = {
            "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
            "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "catalogue_coverage": catalogue_coverage(recommendation_ids, int(eligible.sum())),
            "cold_item_share": cold_count / total_count if total_count else 0.0,
            "evaluated_users": float(len(users)),
        }
    return results
