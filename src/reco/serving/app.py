"""FastAPI application factory."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from reco.logging import request_log
from reco.serving.metrics import ApiMetrics
from reco.serving.schemas import RecommendationRequest, RecommendationResponse, RecommendedItem
from reco.serving.store import BundleStore


def _home(value: Path | None) -> Path:
    if value:
        return value.resolve()
    return Path(os.environ.get("RECO_HOME", "var")).expanduser().resolve()


def create_app(home: Path | None = None) -> FastAPI:
    """Create an API bound to one mutable bundle store."""
    store = BundleStore(_home(home))
    metrics = ApiMetrics.create()
    app = FastAPI(title="Netflix Recommender", version="1.0.0")
    app.state.store = store
    app.state.metrics = metrics

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        store.reload_if_changed()
        if store.get() is None:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "error": store.error})
        return {"status": "ready", "model_version": store.version}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model")
    def model_manifest() -> dict[str, Any]:
        store.reload_if_changed()
        engine = store.get()
        if engine is None:
            raise HTTPException(status_code=503, detail="Model bundle is not loaded")
        return engine.bundle.manifest

    @app.post("/v1/recommendations", response_model=RecommendationResponse)
    def recommendations(request: RecommendationRequest) -> RecommendationResponse:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        store.reload_if_changed()
        engine = store.get()
        if engine is None:
            metrics.requests.labels(status="503").inc()
            raise HTTPException(status_code=503, detail="Model bundle is not loaded")
        strategy, items = engine.recommend(request.user_id, request.limit, request.content_type)
        version = store.version or "unknown"
        metrics.requests.labels(status="200").inc()
        metrics.strategies.labels(strategy=strategy).inc()
        metrics.model_version.labels(version=version).set(1)
        metrics.latency.observe(time.perf_counter() - started)
        request_log(
            request_id=request_id,
            user_id=request.user_id,
            model_version=version,
            strategy=strategy,
            item_ids=[item["item_id"] for item in items],
        )
        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            strategy=strategy,
            model_version=version,
            items=[RecommendedItem(**item) for item in items],
        )

    return app


app = create_app()
