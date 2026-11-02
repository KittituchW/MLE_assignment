"""Pinned, checksummed catalogue ingestion."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from loguru import logger

from reco.config import runtime_paths


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    logger.info("Downloading {}", url)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def ingest(cfg: Any, source: str | Path | None = None, home: Path | None = None) -> dict[str, Path]:
    """Fetch or copy both raw files, verify hashes, and return their paths.

    Parameters
    ----------
    cfg:
        Resolved application configuration.
    source:
        Optional directory containing the two configured filenames.
    home:
        Runtime root override, primarily for tests.
    """
    raw = runtime_paths(home)["raw"]
    source_path = Path(source).expanduser().resolve() if source else None
    outputs: dict[str, Path] = {}
    for key in ("movies", "tv"):
        item = cfg.data[key]
        destination = raw / str(item.filename)
        expected = str(item.sha256)
        if destination.exists() and sha256_file(destination) == expected:
            logger.info("ingest skip: {} already verified", destination.name)
        else:
            if source_path:
                origin = source_path / str(item.filename) if source_path.is_dir() else source_path
                if not origin.is_file():
                    raise FileNotFoundError(f"Missing local source file: {origin}")
                shutil.copy2(origin, destination)
            else:
                _download(str(item.url), destination)
            actual = sha256_file(destination)
            if actual != expected:
                destination.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {item.filename}: expected {expected}, got {actual}")
        outputs[key] = destination
    return outputs
