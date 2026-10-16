"""Tests for the FastAPI layer using TestClient."""

from fastapi.testclient import TestClient
from src.api.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_without_index():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "hello"})
    # No index loaded at import time -> 400
    assert response.status_code == 400


def test_ingest_empty():
    client = TestClient(app)
    response = client.post("/ingest", json={"documents": []})
    assert response.status_code == 400
