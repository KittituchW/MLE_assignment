"""Model-matrix construction shared by evaluation and bundling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from reco.models.base import minmax
from reco.models.content import ContentRecommender, fit_content_embeddings
from reco.models.hybrid import HybridRecommender
from reco.models.item_knn import ItemKNNRecommender, fit_item_similarity
from reco.models.popularity import PopularityRecommender


@dataclass
class TrainedModels:
    """All matrices, indexes, and model objects required by evaluation/bundling."""

    catalog: pd.DataFrame
    item_ids: list[str]
    user_ids: list[str]
    history: sparse.csr_matrix
    similarity: sparse.csr_matrix
    embeddings: np.ndarray
    popularity: np.ndarray
    models: dict[str, Any]


def build_history(
    interactions: pd.DataFrame, item_ids: list[str], user_ids: list[str] | None = None
) -> tuple[sparse.csr_matrix, list[str]]:
    """Build a watch-weighted user-item matrix with deterministic indexes."""
    users = user_ids or sorted(interactions.user_id.astype(str).unique().tolist())
    user_index = {value: index for index, value in enumerate(users)}
    item_index = {value: index for index, value in enumerate(item_ids)}
    filtered = interactions[
        interactions.user_id.astype(str).isin(user_index) & interactions.item_id.astype(str).isin(item_index)
    ]
    rows = filtered.user_id.astype(str).map(user_index).to_numpy()
    columns = filtered.item_id.astype(str).map(item_index).to_numpy()
    values = filtered.watch_ratio.to_numpy(dtype=np.float32)
    history = sparse.coo_matrix((values, (rows, columns)), shape=(len(users), len(item_ids)), dtype=np.float32).tocsr()
    history.sum_duplicates()
    history.data = np.clip(history.data, 0, 1)
    return history, users


def popularity_scores(catalog: pd.DataFrame, history: sparse.csr_matrix | None = None) -> np.ndarray:
    """Blend interaction prevalence with Bayesian quality and catalogue popularity."""
    metadata = 0.55 * minmax(
        catalog.bayesian_rating.to_numpy(dtype=np.float32)
    ) + 0.45 * catalog.popularity_percentile.to_numpy(dtype=np.float32)
    if history is None:
        return metadata.astype(np.float32)
    interaction_prevalence = minmax(np.asarray(history.getnnz(axis=0)).ravel().astype(np.float32))
    return (0.35 * interaction_prevalence + 0.65 * metadata).astype(np.float32)


def train_models(catalog: pd.DataFrame, interactions: pd.DataFrame, cfg: Any) -> TrainedModels:
    """Fit the three component models and configured hybrid."""
    ordered = catalog.sort_values("item_id", kind="stable").reset_index(drop=True)
    item_ids = ordered.item_id.astype(str).tolist()
    history, user_ids = build_history(interactions, item_ids)
    similarity = fit_item_similarity(history, int(cfg.model.item_knn_neighbors))
    embeddings = fit_content_embeddings(ordered, int(cfg.model.content_components), int(cfg.seed))
    popularity = popularity_scores(ordered, history)
    pop_model = PopularityRecommender(popularity)
    knn_model = ItemKNNRecommender(similarity, history)
    content_model = ContentRecommender(embeddings, history)
    weights = cfg.model.hybrid_weights
    hybrid = HybridRecommender(
        knn_model,
        content_model,
        pop_model,
        (float(weights.item_knn), float(weights.content), float(weights.popularity)),
    )
    return TrainedModels(
        catalog=ordered,
        item_ids=item_ids,
        user_ids=user_ids,
        history=history,
        similarity=similarity,
        embeddings=embeddings,
        popularity=popularity,
        models={"popularity": pop_model, "item_knn": knn_model, "content": content_model, "hybrid": hybrid},
    )
