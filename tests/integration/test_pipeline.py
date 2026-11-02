from pathlib import Path

import pandas as pd

from reco.bundle.pipeline import evaluate_bundle, train_bundle
from reco.bundle.reader import load_bundle
from reco.config import load_config, runtime_paths
from reco.data.ingest import ingest
from reco.data.prepare import prepare
from reco.data.validate import validate
from reco.interactions.simulator import SimulatedInteractionSource, write_simulated
from reco.interactions.source import ParquetInteractionSource


def _fixture_config():
    cfg = load_config()
    cfg.data.movies.sha256 = "72cf650bc9f3f1e01fb29ec5b069720e3ed93920ed803314a4022a67527c5b8a"
    cfg.data.tv.sha256 = "636e663e119792b58c1710c469b8627d5ccd1e75d731b0723ad934a5c38e7b50"
    cfg.simulator.users = 30
    cfg.simulator.fixed_demo_events = 20
    cfg.simulator.max_events = 50
    cfg.model.item_knn_neighbors = 10
    cfg.model.content_components = 16
    cfg.evaluation.max_users = 20
    return cfg


def test_full_fixture_pipeline_and_parquet_source(tmp_path: Path) -> None:
    cfg = _fixture_config()
    home = tmp_path / "reco-home"
    source = Path(__file__).parents[1] / "fixtures" / "raw"
    outputs = ingest(cfg, source=source, home=home)
    assert set(outputs) == {"movies", "tv"}
    assert validate(cfg, home) == {"movies": 150, "tv": 150}
    catalog_path = prepare(cfg, home)
    catalog = pd.read_parquet(catalog_path)
    assert len(catalog) == 300
    assert {"movie_18823", "tv_18823"} <= set(catalog.item_id)

    paths = runtime_paths(home)
    events_path = write_simulated(
        SimulatedInteractionSource(catalog, cfg), paths["interactions"] / "interactions.parquet"
    )
    events = ParquetInteractionSource(events_path).load()
    assert len(events) > 300
    assert events.user_id.nunique() == 30

    bundle_path = train_bundle(cfg, home, version="fixture-v1")
    report = evaluate_bundle(cfg, home, version="fixture-v1")
    loaded = load_bundle(bundle_path)
    assert loaded.manifest["interaction_source"] == "simulated"
    assert report["models"]["hybrid"]["evaluated_users"] > 0


def test_ingest_is_idempotent_and_rejects_bad_checksum(tmp_path: Path) -> None:
    cfg = _fixture_config()
    source = Path(__file__).parents[1] / "fixtures" / "raw"
    first = ingest(cfg, source=source, home=tmp_path)
    second = ingest(cfg, source=source, home=tmp_path)
    assert first == second
    first["movies"].write_text("corrupt", encoding="utf-8")
    cfg.data.movies.sha256 = "0" * 64
    try:
        ingest(cfg, source=source, home=tmp_path)
    except ValueError as exc:
        assert "Checksum mismatch" in str(exc)
    else:
        raise AssertionError("bad checksum was accepted")
