from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy import sparse

from reco.bundle.writer import write_bundle, write_pointer
from reco.config import load_config
from reco.models.training import TrainedModels
from reco.serving.app import create_app


def _bundle(home: Path, version: str, title: str = "Second") -> Path:
    catalog = pd.DataFrame(
        [
            {
                "item_id": f"movie_{index}",
                "title": "Watched" if index == 0 else (title if index == 1 else f"Title {index}"),
                "normalized_title": f"title {index}",
                "content_type": "movie" if index < 5 else "tv",
                "genres": ["Action", f"G{index}"],
                "release_year": 2020 + index,
                "recommendable": True,
            }
            for index in range(8)
        ]
    )
    history = sparse.csr_matrix([[1, 0.8, 0.7, 0, 0, 0, 0, 0]], dtype=np.float32)
    similarity = sparse.csr_matrix(np.ones((8, 8), dtype=np.float32) - np.eye(8, dtype=np.float32))
    trained = TrainedModels(
        catalog,
        catalog.item_id.tolist(),
        ["A"],
        history,
        similarity,
        np.eye(8, dtype=np.float32),
        np.linspace(0, 1, 8, dtype=np.float32),
        {},
    )
    source = home / f"{version}.input"
    source.write_text("x", encoding="utf-8")
    return write_bundle(trained, load_config(), version, [source], home)


def test_api_contract_fallback_metrics_and_validation(tmp_path: Path) -> None:
    _bundle(tmp_path, "v1")
    write_pointer(tmp_path / "artifacts" / "CHAMPION", "v1")
    client = TestClient(create_app(tmp_path))
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json()["model_version"] == "v1"
    known = client.post("/v1/recommendations", json={"user_id": "A", "limit": 4})
    assert known.status_code == 200
    assert known.json()["strategy"] == "personalized"
    assert len(known.json()["items"]) == 4
    assert all("Because you watched" not in item["reason"] for item in known.json()["items"])
    unknown = client.post("/v1/recommendations", json={"user_id": "missing"})
    assert unknown.status_code == 200
    assert unknown.json()["strategy"] == "cold_start_popular"
    assert client.post("/v1/recommendations", json={"user_id": "A", "limit": 51}).status_code == 422
    assert "reco_requests_total" in client.get("/metrics").text
    assert client.get("/v1/model").json()["version"] == "v1"


def test_missing_bundle_is_not_ready(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/healthz").status_code == 200
    readiness = client.get("/readyz")
    assert readiness.status_code == 503
    assert "error" not in readiness.json()["detail"]
    assert client.post("/v1/recommendations", json={"user_id": "A"}).status_code == 503


def test_hot_swap_and_failed_swap_keep_old_model(tmp_path: Path) -> None:
    _bundle(tmp_path, "v1")
    _bundle(tmp_path, "v2", title="Changed")
    pointer = tmp_path / "artifacts" / "CHAMPION"
    write_pointer(pointer, "v1")
    app = create_app(tmp_path)
    client = TestClient(app)
    assert client.get("/v1/model").json()["version"] == "v1"
    write_pointer(pointer, "v2")
    assert client.get("/v1/model").json()["version"] == "v2"
    write_pointer(pointer, "corrupt")
    assert client.get("/v1/model").json()["version"] == "v2"
    assert app.state.store.error is not None
    first_error = app.state.store.error
    assert client.get("/readyz").json()["model_version"] == "v2"
    assert app.state.store.error == first_error
