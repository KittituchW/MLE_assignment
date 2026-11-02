from fastapi.testclient import TestClient

from discovery_reco.api import app
from discovery_reco.model import train


def test_api_contract(catalog, events):
    app.state.model = train(catalog, events)
    client = TestClient(app, raise_server_exceptions=True)
    response = client.post("/v1/recommendations", json={"user_id": "A", "limit": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "behavior_content_profile"
    assert [item["rank"] for item in body["items"]] == [1, 2, 3]
