from pathlib import Path

from reco.config import config_hash, load_config, project_root, runtime_paths


def test_config_inheritance_and_stable_hash() -> None:
    cfg = load_config("configs/dev.yaml")
    assert cfg.simulator.users == 300
    assert cfg.data.movies.filename.endswith(".csv")
    assert config_hash(cfg) == config_hash(cfg)


def test_runtime_paths(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path)
    assert paths["raw"].is_dir()
    assert paths["champion"] == tmp_path / "artifacts" / "CHAMPION"


def test_project_root_override_loads_container_config(tmp_path: Path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("RECO_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    assert load_config().seed == 7
