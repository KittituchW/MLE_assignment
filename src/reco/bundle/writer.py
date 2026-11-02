"""Training-side portable bundle writer."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from reco.bundle.manifest import make_manifest
from reco.config import project_root, runtime_paths
from reco.models.training import TrainedModels


def write_pointer(path: Path, value: str) -> None:
    """Atomically write a Windows-compatible pointer file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.strip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _catalog_records(trained: TrainedModels) -> list[dict[str, Any]]:
    return [
        {
            "item_id": str(row.item_id),
            "title": str(row.title),
            "normalized_title": str(row.normalized_title),
            "content_type": str(row.content_type),
            "genres": [str(genre) for genre in row.genres],
            "release_year": int(row.release_year),
            "recommendable": bool(row.recommendable),
        }
        for row in trained.catalog.itertuples(index=False)
    ]


def write_bundle(
    trained: TrainedModels,
    cfg: Any,
    version: str,
    data_paths: list[Path],
    home: Path | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write a complete immutable bundle, returning an existing valid version idempotently."""
    paths = runtime_paths(home)
    destination = paths["models"] / version
    if (destination / "manifest.json").is_file():
        write_pointer(paths["models"].parent / "CANDIDATE", version)
        return destination
    temporary = paths["models"] / f".{version}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    manifest = make_manifest(cfg, project_root(), version, data_paths, metrics)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "catalog.json").write_text(
        json.dumps(_catalog_records(trained), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (temporary / "users.json").write_text(
        json.dumps(trained.user_ids, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (temporary / "items.json").write_text(
        json.dumps(trained.item_ids, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    sparse.save_npz(temporary / "similarity.npz", trained.similarity, compressed=True)
    sparse.save_npz(temporary / "history.npz", trained.history, compressed=True)
    np.save(temporary / "embeddings.npy", trained.embeddings, allow_pickle=False)
    np.save(temporary / "popularity.npy", trained.popularity, allow_pickle=False)
    (temporary / "eval_report.json").write_text(
        json.dumps(metrics or {}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    write_pointer(paths["models"].parent / "CANDIDATE", version)
    return destination


def update_evaluation(bundle_path: Path, metrics: dict[str, Any]) -> None:
    """Atomically update the evaluation report and manifest metrics."""
    report = bundle_path / "eval_report.json"
    report_tmp = report.with_suffix(".json.tmp")
    report_tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(report_tmp, report)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"] = metrics
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
