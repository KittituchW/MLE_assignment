"""Failure-safe champion pointer polling and hot swap."""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from reco.bundle.reader import BundleLoadError, load_bundle
from reco.serving.engine import RecommendationEngine


class BundleStore:
    """Keep an immutable serving snapshot and avoid retrying broken pointers per request."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.pointer = self.home / "artifacts" / "CHAMPION"
        self._lock = threading.Lock()
        self._version: str | None = None
        self._engine: RecommendationEngine | None = None
        self._error: str | None = None
        self._rejected_version: str | None = None
        self.reload_if_changed(force=True)

    @property
    def version(self) -> str | None:
        """Return the loaded model version."""
        return self.snapshot()[1]

    @property
    def error(self) -> str | None:
        """Return the latest load error, if any."""
        return self._error

    def get(self) -> RecommendationEngine | None:
        """Return the current immutable engine reference."""
        return self.snapshot()[0]

    def snapshot(self) -> tuple[RecommendationEngine | None, str | None]:
        """Read the engine and its matching version atomically."""
        with self._lock:
            return self._engine, self._version

    def reload_if_changed(self, force: bool = False) -> bool:
        """Try a new pointer once and keep the last good model on load failure."""
        with self._lock:
            try:
                if not self.pointer.is_file():
                    raise BundleLoadError(f"Champion pointer missing: {self.pointer}")
                version = self.pointer.read_text(encoding="utf-8").strip()
                if not version:
                    raise BundleLoadError("Champion pointer is empty")
                if not force and (version == self._version or version == self._rejected_version):
                    return False
                try:
                    bundle = load_bundle(self.home / "artifacts" / "models" / version)
                    candidate = RecommendationEngine(bundle)
                except Exception:
                    self._rejected_version = version
                    raise
                self._engine = candidate
                self._version = version
                self._rejected_version = None
                self._error = None
                logger.info("loaded model bundle {}", version)
                return True
            except Exception as exc:
                self._error = str(exc)
                logger.error("model bundle reload failed; retaining version {}: {}", self._version, exc)
                return False
