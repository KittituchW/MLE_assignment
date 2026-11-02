"""Per-app Prometheus collectors."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


@dataclass(frozen=True)
class ApiMetrics:
    """Metrics registered in an isolated registry for testability."""

    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    strategies: Counter
    model_version: Gauge

    @classmethod
    def create(cls) -> ApiMetrics:
        """Create all collectors."""
        registry = CollectorRegistry()
        return cls(
            registry,
            Counter("reco_requests_total", "Recommendation requests", ["status"], registry=registry),
            Histogram("reco_request_latency_seconds", "Recommendation latency", registry=registry),
            Counter("reco_strategy_total", "Responses by strategy", ["strategy"], registry=registry),
            Gauge("reco_model_version", "Currently loaded model", ["version"], registry=registry),
        )
