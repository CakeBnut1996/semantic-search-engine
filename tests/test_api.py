import pytest
from fastapi.testclient import TestClient
from seed.api.server import app
from seed.config import AppConfig

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/v1/search/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "dependencies" in data

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SEED Search Gateway is active."}

# Note: /v1/search/semantic requires valid Solr/LLM mock or connection
# For unit tests, we should mock the retriever and generator.
