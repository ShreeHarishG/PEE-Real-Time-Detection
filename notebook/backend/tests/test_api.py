import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_zones():
    response = client.get("/api/v1/zones")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_cameras():
    response = client.get("/api/v1/cameras")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
