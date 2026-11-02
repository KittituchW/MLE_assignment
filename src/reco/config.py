"""Configuration and runtime-path helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def project_root() -> Path:
    """Return the repository root independent of the current directory."""
    return Path(__file__).resolve().parents[2]


def reco_home() -> Path:
    """Return and create the mutable runtime root."""
    root = Path(os.environ.get("RECO_HOME", project_root() / "var")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_config(path: str | Path | None = None) -> DictConfig:
    """Load a YAML configuration, resolving the small inheritance convention."""
    config_path = Path(path) if path else project_root() / "configs" / "base.yaml"
    if not config_path.is_absolute():
        config_path = (project_root() / config_path).resolve()
    cfg = OmegaConf.load(config_path)
    parent = cfg.pop("defaults", None)
    if parent:
        parent_path = config_path.with_name(f"{parent}.yaml")
        cfg = OmegaConf.merge(OmegaConf.load(parent_path), cfg)
    OmegaConf.resolve(cfg)
    return cfg


def config_hash(cfg: DictConfig) -> str:
    """Return a stable SHA-256 for a resolved configuration."""
    plain: Any = OmegaConf.to_container(cfg, resolve=True)
    payload = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def runtime_paths(home: Path | None = None) -> dict[str, Path]:
    """Return named runtime paths and create their parent directories."""
    root = home or reco_home()
    paths = {
        "home": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "interactions": root / "data" / "interactions",
        "models": root / "artifacts" / "models",
        "champion": root / "artifacts" / "CHAMPION",
        "mlflow_db": root / "mlflow.db",
    }
    for key in ("raw", "processed", "interactions", "models"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths

