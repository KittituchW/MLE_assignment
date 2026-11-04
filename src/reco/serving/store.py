"""Failure-safe, non-blocking champion pointer polling and hot swap."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger

from reco.bundle.reader import BundleLoadError, load_bundle
from reco.serving.engine import RecommendationEngine


class BundleStore:
    """Keep a last-known-good serving snapshot while loading a replacement."""

    def __init__(self, home: Path, poll_seconds: float = 5.0) -> None:
        if poll_seconds < 0:
            raise ValueError("poll_seconds cannot be negative")
        self.home = Path(home)
        self.pointer = self.home / "artifacts" / "CHAMPION"
        self._state_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._poll_seconds = poll_seconds
        self._next_poll_at = 0.0
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
        with self._state_lock:
            return self._error

    def get(self) -> RecommendationEngine | None:
        """Return the current immutable engine reference."""
        return self.snapshot()[0]

    def snapshot(self) -> tuple[RecommendationEngine | None, str | None]:
        """Read the matching engine/version pair without waiting for bundle I/O."""
        with self._state_lock:
            return self._engine, self._version

    def reload_if_changed(self, force: bool = False) -> bool:
        """Check at a bounded interval; keep serving during a slow/failed reload."""
        if not force and time.monotonic() < self._next_poll_at:
            return False
        if not self._load_lock.acquire(blocking=False):
            return False
        try:
            if not force and time.monotonic() < self._next_poll_at:
                return False
            self._next_poll_at = time.monotonic() + self._poll_seconds
            version: str | None = None
            try:
                if not self.pointer.is_file():
                    raise BundleLoadError(f"Champion pointer missing: {self.pointer}")
                version = self.pointer.read_text(encoding="utf-8").strip()
                if not version:
                    raise BundleLoadError("Champion pointer is empty")
                with self._state_lock:
                    if not force and (version == self._version or version == self._rejected_version):
                        return False

                # Deliberately outside the state lock: existing requests retain the old engine.
                bundle = load_bundle(self.home / "artifacts" / "models" / version)
                candidate = RecommendationEngine(bundle)
                with self._state_lock:
                    self._engine = candidate
                    self._version = version
                    self._rejected_version = None
                    self._error = None
                logger.info("loaded model bundle {}", version)
                return True
            except Exception as exc:
                with self._state_lock:
                    if version:
                        self._rejected_version = version
                    self._error = str(exc)
                    active_version = self._version
                logger.error("model bundle reload failed; retaining version {}: {}", active_version, exc)
                return False
        finally:
            self._load_lock.release()
