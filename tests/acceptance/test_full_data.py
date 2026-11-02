from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reco.serving.app import create_app


@pytest.mark.slow
def test_assignment_given_when_then_on_full_bundle() -> None:
    home = Path("var").resolve()
    if not (home / "artifacts" / "CHAMPION").is_file():
        pytest.skip("run-all has not created the full-data champion")
    client = TestClient(create_app(home))
    action = client.post("/v1/recommendations", json={"user_id": "A"})
    assert action.status_code == 200
    assert len(action.json()["items"]) == 10
    assert sum("Action" in item["genres"] for item in action.json()["items"]) >= 6
    romance = client.post("/v1/recommendations", json={"user_id": "B"})
    assert romance.status_code == 200
    assert sum("Romance" in item["genres"] for item in romance.json()["items"]) >= 6
    unknown = client.post("/v1/recommendations", json={"user_id": "never-seen"})
    assert unknown.status_code == 200
    assert unknown.json()["strategy"] == "cold_start_popular"
