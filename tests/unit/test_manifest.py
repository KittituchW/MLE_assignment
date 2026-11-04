"""Regression tests for source provenance in a Git-free runtime image."""

import subprocess
from pathlib import Path

from reco.bundle.manifest import git_sha


def test_missing_git_is_not_fatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RECO_GIT_SHA", raising=False)

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(subprocess, "run", missing_git)
    assert git_sha(tmp_path) == "unknown"


def test_explicit_source_sha_works_without_git(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RECO_GIT_SHA", "abcdef1234567890")
    assert git_sha(tmp_path) == "abcdef1234567890"
