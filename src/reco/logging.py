"""Structured logging setup."""

from __future__ import annotations

import json
import sys
from typing import Any

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure JSON logs for command-line stages and the API."""
    logger.remove()
    logger.add(sys.stderr, level=level, serialize=True, backtrace=False, diagnose=False)


def request_log(**fields: Any) -> None:
    """Write the single structured exposure line required per API request."""
    logger.bind(event="recommendation_exposure").info(json.dumps(fields, separators=(",", ":"), default=str))
