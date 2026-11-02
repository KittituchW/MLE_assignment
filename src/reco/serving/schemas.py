"""Validated HTTP request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendationRequest(BaseModel):
    """Recommendation query."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=10, ge=1, le=50)
    content_type: Literal["movie", "tv"] | None = None


class RecommendedItem(BaseModel):
    """One ranked catalogue result."""

    rank: int
    item_id: str
    title: str
    content_type: Literal["movie", "tv"]
    genres: list[str]
    release_year: int
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    """Versioned recommendation response."""

    request_id: str
    user_id: str
    strategy: Literal["personalized", "content_based", "cold_start_popular", "degraded_popular"]
    model_version: str
    items: list[RecommendedItem]
