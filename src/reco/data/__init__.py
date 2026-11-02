"""Catalogue ingestion, validation, preparation, and policy."""

from reco.data.ingest import ingest
from reco.data.prepare import prepare
from reco.data.validate import validate

__all__ = ["ingest", "prepare", "validate"]
