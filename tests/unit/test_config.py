from pathlib import Path

from reco.config import config_hash, load_config, runtime_paths


def test_config_inheritance_and_stable_hash() -> None:
    cfg = load_config("configs/dev.yaml")
    assert cfg.simulator.users == 300
    assert cfg.data.movies.filename.endswith(".csv")
    assert config_hash(cfg) == config_hash(cfg)


def test_runtime_paths(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path)
    assert paths["raw"].is_dir()
    assert paths["champion"] == tmp_path / "artifacts" / "CHAMPION"
