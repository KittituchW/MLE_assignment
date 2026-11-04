from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from reco.bundle.reader import BundleLoadError, load_bundle
from reco.bundle.registry import PromotionRejected, promote_bundle
from reco.bundle.writer import update_evaluation, write_bundle
from reco.config import load_config
from reco.models.training import TrainedModels


def _trained() -> TrainedModels:
    catalog = pd.DataFrame(
        [
            {
                "item_id": "movie_1",
                "title": "One",
                "normalized_title": "one",
                "content_type": "movie",
                "genres": ["Action"],
                "release_year": 2020,
                "recommendable": True,
            },
            {
                "item_id": "tv_2",
                "title": "Two",
                "normalized_title": "two",
                "content_type": "tv",
                "genres": ["Drama"],
                "release_year": 2021,
                "recommendable": True,
            },
        ]
    )
    history = sparse.csr_matrix([[1, 0]], dtype=np.float32)
    similarity = sparse.csr_matrix([[0, 0.5], [0.5, 0]], dtype=np.float32)
    return TrainedModels(
        catalog,
        ["movie_1", "tv_2"],
        ["A"],
        history,
        similarity,
        np.eye(2, dtype=np.float32),
        np.array([1, 0.5], dtype=np.float32),
        {},
    )


def test_bundle_round_trip_and_corrupt_shape(tmp_path: Path) -> None:
    data = tmp_path / "input.bin"
    data.write_text("x", encoding="utf-8")
    bundle = write_bundle(_trained(), load_config(), "v1", [data], tmp_path)
    loaded = load_bundle(bundle)
    assert loaded.manifest["version"] == "v1"
    assert loaded.user_ids == ["A"]
    np.save(bundle / "popularity.npy", np.array([1], dtype=np.float32), allow_pickle=False)
    with pytest.raises(BundleLoadError, match="shape"):
        load_bundle(bundle)


def _report(recall: float, passed: bool = True) -> dict:
    values = {"recall_at_10": recall, "ndcg_at_10": recall}
    return {
        "models": {"hybrid": values, "popularity": {"recall_at_10": 0.01, "ndcg_at_10": 0.01}},
        "promotion_gate": {"passed": passed, "reasons": [] if passed else ["failed"]},
    }


def test_failed_gate_leaves_champion_and_rollback_works(tmp_path: Path) -> None:
    data = tmp_path / "input.bin"
    data.write_text("x", encoding="utf-8")
    first = write_bundle(_trained(), load_config(), "v1", [data], tmp_path)
    update_evaluation(first, _report(0.03))
    promote_bundle(tmp_path, "v1", track_mlflow=False)
    second = write_bundle(_trained(), load_config(), "v2", [data], tmp_path)
    update_evaluation(second, _report(0.001, passed=False))
    with pytest.raises(PromotionRejected):
        promote_bundle(tmp_path, "v2", track_mlflow=False)
    champion = tmp_path / "artifacts" / "CHAMPION"
    assert champion.read_text().strip() == "v1"
    promote_bundle(tmp_path, "v2", track_mlflow=False, force=True)
    assert champion.read_text().strip() == "v2"
    promote_bundle(tmp_path, "v1", track_mlflow=False, force=True)
    assert champion.read_text().strip() == "v1"


def test_explicit_candidate_version_does_not_bypass_gate(tmp_path: Path, monkeypatch) -> None:
    from argparse import Namespace

    from reco.cli import _promote

    data = tmp_path / "input.bin"
    data.write_text("x", encoding="utf-8")
    candidate = write_bundle(_trained(), load_config(), "rejected", [data], tmp_path)
    update_evaluation(candidate, _report(0.0, passed=False))
    monkeypatch.setenv("RECO_HOME", str(tmp_path))
    with pytest.raises(PromotionRejected):
        _promote(Namespace(version="rejected", force=False))
