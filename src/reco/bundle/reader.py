"""Strict serving-safe bundle reader (JSON/NPZ/NPY only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from reco.bundle.manifest import SCHEMA_VERSION


class BundleLoadError(RuntimeError):
    """Raised when a model bundle is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class BundleData:
    """Validated portable arrays and metadata."""

    path: Path
    manifest: dict[str, Any]
    catalog: list[dict[str, Any]]
    user_ids: list[str]
    item_ids: list[str]
    similarity: sparse.csr_matrix
    history: sparse.csr_matrix
    embeddings: np.ndarray
    popularity: np.ndarray


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_bundle(path: str | Path) -> BundleData:
    """Load and validate one bundle without pickle or training libraries."""
    root = Path(path)
    required = {
        "manifest.json",
        "catalog.json",
        "users.json",
        "items.json",
        "similarity.npz",
        "history.npz",
        "embeddings.npy",
        "popularity.npy",
    }
    missing = [name for name in sorted(required) if not (root / name).is_file()]
    if missing:
        raise BundleLoadError(f"Bundle missing files: {missing}")
    try:
        manifest = _json(root / "manifest.json")
        catalog = _json(root / "catalog.json")
        user_ids = _json(root / "users.json")
        item_ids = _json(root / "items.json")
        similarity = sparse.load_npz(root / "similarity.npz").tocsr().astype(np.float32)
        history = sparse.load_npz(root / "history.npz").tocsr().astype(np.float32)
        embeddings = np.load(root / "embeddings.npy", allow_pickle=False).astype(np.float32, copy=False)
        popularity = np.load(root / "popularity.npy", allow_pickle=False).astype(np.float32, copy=False)
    except Exception as exc:
        raise BundleLoadError(f"Could not load bundle {root}: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleLoadError(f"Unsupported schema version {manifest.get('schema_version')}; expected {SCHEMA_VERSION}")
    item_count = len(item_ids)
    user_count = len(user_ids)
    if len(catalog) != item_count:
        raise BundleLoadError("Catalogue and item index lengths differ")
    if similarity.shape != (item_count, item_count):
        raise BundleLoadError("Similarity shape does not match item index")
    if history.shape != (user_count, item_count):
        raise BundleLoadError("History shape does not match indexes")
    if embeddings.shape[0] != item_count or popularity.shape != (item_count,):
        raise BundleLoadError("Dense array shape does not match item index")
    if [str(record.get("item_id")) for record in catalog] != item_ids:
        raise BundleLoadError("Catalogue order does not match item index")
    return BundleData(root, manifest, catalog, user_ids, item_ids, similarity, history, embeddings, popularity)
