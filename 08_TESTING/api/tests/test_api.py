import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_cameras():
    response = client.get("/api/v1/cameras")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_zones():
    response = client.get("/api/v1/zones")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_camera():
    payload = {
        "name": "Test Camera 1",
        "source_type": "rtsp",
        "rtsp_url": "rtsp://localhost/test",
        "zone_id": 1,
        "is_active": True
    }
    response = client.post("/api/v1/cameras", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == payload["name"]
    assert "id" in data

def test_get_stats():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "active_violations" in data
    assert "acknowledged" in data
