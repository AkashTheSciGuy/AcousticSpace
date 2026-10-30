"""Smoke tests for AcousticSpace's initial FastAPI endpoints.

Run these from backend/ with: python -m pytest -q
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_welcome_message(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to AcousticSpace API",
        "status": "running",
    }


def test_health_returns_service_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "AcousticSpace Backend",
    }


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/not-a-route")

    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"


def test_openapi_documents_health_route(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
