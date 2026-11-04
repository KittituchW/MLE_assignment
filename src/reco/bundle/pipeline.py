"""Independent train and evaluate bundle stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from reco.bundle.manifest import make_version
from reco.bundle.reader import BundleData, load_bundle
from reco.bundle.writer import update_evaluation, write_bundle
from reco.config import project_root, runtime_paths
from reco.evaluation.evaluate import evaluate_models
from reco.evaluation.gate import evaluate_gate
from reco.evaluation.split import temporal_split
from reco.models.content import ContentRecommender
from reco.models.hybrid import HybridRecommender
from reco.models.item_knn import ItemKNNRecommender
from reco.models.popularity import PopularityRecommender
from reco.models.training import TrainedModels, train_models
from reco.serving.engine import RecommendationEngine


def candidate_version(home: Path | None = None) -> str:
    """Read the current candidate pointer."""
    path = runtime_paths(home)["models"].parent / "CANDIDATE"
    if not path.is_file():
        raise FileNotFoundError("No candidate pointer; run train first")
    return path.read_text(encoding="utf-8").strip()


def train_bundle(cfg: Any, home: Path | None = None, version: str | None = None) -> Path:
    """Fit on all pre-test events and write an unevaluated candidate bundle."""
    paths = runtime_paths(home)
    catalog_path = paths["processed"] / "catalog.parquet"
    interactions_path = paths["interactions"] / "interactions.parquet"
    catalog = pd.read_parquet(catalog_path)
    interactions = pd.read_parquet(interactions_path)
    split = temporal_split(
        interactions,
        str(cfg.as_of_date),
        int(cfg.evaluation.holdout_days),
        int(cfg.evaluation.validation_days),
    )
    timestamps = pd.to_datetime(interactions.event_ts, utc=True)
    pretest = interactions.loc[timestamps < split.test_start]
    trained = train_models(catalog, pretest, cfg)
    selected_version = version or make_version(cfg, project_root())
    return write_bundle(trained, cfg, selected_version, [catalog_path, interactions_path], home)


def reconstruct(bundle: BundleData, cfg: Any) -> TrainedModels:
    """Reconstruct training model wrappers from portable arrays."""
    catalog = pd.DataFrame(bundle.catalog)
    pop_model = PopularityRecommender(bundle.popularity)
    knn_model = ItemKNNRecommender(bundle.similarity, bundle.history)
    content_model = ContentRecommender(bundle.embeddings, bundle.history)
    weights = bundle.manifest.get("serving", {}).get(
        "hybrid_weights",
        [
            float(cfg.model.hybrid_weights.item_knn),
            float(cfg.model.hybrid_weights.content),
            float(cfg.model.hybrid_weights.popularity),
        ],
    )
    hybrid = HybridRecommender(knn_model, content_model, pop_model, tuple(float(w) for w in weights))
    return TrainedModels(
        catalog,
        bundle.item_ids,
        bundle.user_ids,
        bundle.history,
        bundle.similarity,
        bundle.embeddings,
        bundle.popularity,
        {"popularity": pop_model, "item_knn": knn_model, "content": content_model, "hybrid": hybrid},
    )


def evaluate_bundle(cfg: Any, home: Path | None = None, version: str | None = None) -> dict[str, Any]:
    """Evaluate an existing candidate, write its report, and return gate status."""
    paths = runtime_paths(home)
    selected_version = version or candidate_version(home)
    bundle_path = paths["models"] / selected_version
    bundle = load_bundle(bundle_path)
    interactions = pd.read_parquet(paths["interactions"] / "interactions.parquet")
    split = temporal_split(
        interactions,
        str(cfg.as_of_date),
        int(cfg.evaluation.holdout_days),
        int(cfg.evaluation.validation_days),
    )
    k = int(cfg.evaluation.k)
    if k != 10:
        raise ValueError("This release supports only evaluation.k=10; metrics and gate are defined at @10")
    diversity_fraction = float(bundle.manifest.get("serving", {}).get("diversity_fraction", 0.4))
    metrics = evaluate_models(
        reconstruct(bundle, cfg),
        split.test,
        int(cfg.evaluation.max_users),
        k,
        serving_engine=RecommendationEngine(bundle),
        diversity_fraction=diversity_fraction,
    )
    baseline_ratio = float(cfg.evaluation.min_baseline_ratio)
    champion_tolerance = float(cfg.evaluation.champion_tolerance)
    passed, reasons = evaluate_gate(
        metrics["hybrid"], metrics["popularity"], baseline_ratio=baseline_ratio
    )
    report: dict[str, Any] = {
        "interaction_source": "simulated",
        "notice": (
            "Metrics recover simulator rules and are regression gates only; they do not estimate real-world quality."
        ),
        "models": metrics,
        "promotion_gate": {
            "passed": passed,
            "reasons": reasons,
            "baseline_ratio": baseline_ratio,
            "champion_tolerance": champion_tolerance,
        },
        "split": {
            "validation_start": split.validation_start.isoformat(),
            "test_start": split.test_start.isoformat(),
            "test_events": len(split.test),
        },
    }
    update_evaluation(bundle_path, report)
    return report


def read_report(bundle_path: Path) -> dict[str, Any]:
    """Read and minimally validate an evaluation report."""
    report = json.loads((bundle_path / "eval_report.json").read_text(encoding="utf-8"))
    if "models" not in report or "promotion_gate" not in report:
        raise ValueError(f"Bundle has no completed evaluation: {bundle_path}")
    return report
