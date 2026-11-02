"""Bundle identity and provenance."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from reco.config import config_hash

SCHEMA_VERSION = 1


def git_sha(project: Path) -> str:
    """Return the current repository SHA, or ``unknown`` outside Git."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_version(cfg: Any, project: Path) -> str:
    """Create a deterministic build-date plus Git-SHA version."""
    build_date = str(cfg.build_date).replace("-", "")
    return f"{build_date}-{git_sha(project)[:7]}"


def hash_files(paths: list[Path]) -> str:
    """Hash ordered file contents for data provenance."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def make_manifest(
    cfg: Any,
    project: Path,
    version: str,
    data_paths: list[Path],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete serializable provenance manifest."""
    return {
        "version": version,
        "git_sha": git_sha(project),
        "data_hash": hash_files(data_paths),
        "config_hash": config_hash(cfg),
        "metrics": metrics or {},
        "schema_version": SCHEMA_VERSION,
        "created_time": f"{cfg.build_date}T00:00:00Z",
        "interaction_source": "simulated",
        "simulated_metrics_notice": (
            "Metrics measure recovery of deterministic simulator rules; they are regression gates, not real-world quality."
        ),
    }
