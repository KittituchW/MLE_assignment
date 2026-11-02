"""Configuration name to model-class registry."""

from reco.models.base import Recommender
from reco.models.content import ContentRecommender
from reco.models.hybrid import HybridRecommender
from reco.models.item_knn import ItemKNNRecommender
from reco.models.popularity import PopularityRecommender

MODEL_REGISTRY: dict[str, type[Recommender]] = {
    "popularity": PopularityRecommender,
    "item_knn": ItemKNNRecommender,
    "content": ContentRecommender,
    "hybrid": HybridRecommender,
}
