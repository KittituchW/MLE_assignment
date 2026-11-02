"""FastAPI serving boundary."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import DiscoveryModel


class Request(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=10, ge=1, le=50)
    content_type: str | None = Field(default=None, pattern="^(movie|tv)$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = os.getenv("RECO_ARTIFACT", "artifacts/model")
    app.state.model = DiscoveryModel.load(artifact)
    yield


app = FastAPI(title="Netflix Discovery EDA", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready() -> dict[str, object]:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "items": len(model.catalog), "users": len(model.users)}


@app.post("/v1/recommendations")
def recommendations(request: Request) -> dict[str, object]:
    strategy, items = app.state.model.recommend(request.user_id, request.limit, request.content_type)
    return {"user_id": request.user_id, "strategy": strategy,
            "items": [{"rank": rank, **item.__dict__} for rank, item in enumerate(items, 1)]}

