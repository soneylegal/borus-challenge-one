"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test root endpoint serves HTML or API info."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    """Test health check returns operational metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_vectors" in data
    assert "version" in data


def test_stats_endpoint(client):
    """Test vector stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "collection_name" in data
    assert "embedding_model" in data


def test_chat_validation_error(client):
    """Test chat query validation when query is too short."""
    response = client.post("/chat", json={"query": "a"})
    assert response.status_code == 422


def test_upload_invalid_file_extension(client):
    """Test PDF upload rejects non-PDF file extensions."""
    response = client.post(
        "/ingest/upload",
        files={"file": ("invalid.txt", b"some plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Apenas arquivos PDF" in response.json()["detail"]
