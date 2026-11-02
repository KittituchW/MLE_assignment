"""Failure-safe champion pointer polling and hot swap."""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from reco.bundle.reader import BundleLoadError, load_bundle
from reco.serving.engine import RecommendationEngine


class BundleStore:
    """Hold one immutable engine and swap only after a new bundle fully validates."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.pointer = self.home / "artifacts" / "CHAMPION"
        self._lock = threading.Lock()
        self._version: str | None = None
        self._engine: RecommendationEngine | None = None
        self._error: str | None = None
        self.reload_if_changed(force=True)

    @property
    def version(self) -> str | None:
        """Return the loaded model version."""
        return self._version

    @property
    def error(self) -> str | None:
        """Return the latest load error, if any."""
        return self._error

    def get(self) -> RecommendationEngine | None:
        """Return the current immutable engine reference."""
        return self._engine

    def reload_if_changed(self, force: bool = False) -> bool:
        """Load the pointer target completely before swapping; retain the old engine on failure."""
        try:
            if not self.pointer.is_file():
                raise BundleLoadError(f"Champion pointer missing: {self.pointer}")
            version = self.pointer.read_text(encoding="utf-8").strip()
            if not version:
                raise BundleLoadError("Champion pointer is empty")
            if not force and version == self._version:
                return False
            bundle = load_bundle(self.home / "artifacts" / "models" / version)
            candidate = RecommendationEngine(bundle)
            with self._lock:
                self._engine = candidate
                self._version = version
                self._error = None
            logger.info("loaded model bundle {}", version)
            return True
        except Exception as exc:
            self._error = str(exc)
            logger.error("model bundle reload failed; retaining version {}: {}", self._version, exc)
            return False
