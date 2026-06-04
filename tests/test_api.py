"""Tests for the FastAPI REST API endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# Skip all API tests if FastAPI is not installed
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.serve import app, model, metrics, df


@pytest.fixture
def client():
    """Create a test client. If model/data not loaded, tests will be skipped."""
    if model is None:
        pytest.skip("Model not loaded — run src/train_model.py first")
    return TestClient(app)


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_reports_model_loaded(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True


class TestBrands:
    def test_list_brands(self, client):
        resp = client.get("/brands")
        assert resp.status_code == 200
        data = resp.json()
        assert "brands" in data
        assert isinstance(data["brands"], list)
        assert len(data["brands"]) > 0

    def test_list_series_for_brand(self, client):
        resp = client.get("/brands/Apple/series")
        assert resp.status_code == 200
        data = resp.json()
        assert "series" in data

    def test_list_series_unknown_brand(self, client):
        resp = client.get("/brands/NonExistentBrand/series")
        assert resp.status_code == 404


class TestPredict:
    def test_predict_valid_input(self, client):
        resp = client.post("/predict", json={
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "storage_gb": 128,
            "condition": "مستعمل - ممتاز",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_price_jd" in data
        assert data["predicted_price_jd"] > 0
        assert "lower_bound_jd" in data
        assert "upper_bound_jd" in data
        assert data["lower_bound_jd"] < data["predicted_price_jd"]
        assert data["upper_bound_jd"] > data["predicted_price_jd"]

    def test_predict_with_premium_features(self, client):
        resp = client.post("/predict", json={
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "storage_gb": 256,
            "condition": "مستعمل - ممتاز",
            "is_pro": True,
            "is_max": True,
            "has_warranty": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_price_jd"] > 0

    def test_predict_invalid_condition(self, client):
        resp = client.post("/predict", json={
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "condition": "invalid_condition",
        })
        assert resp.status_code == 422  # Validation error

    def test_predict_unknown_brand(self, client):
        resp = client.post("/predict", json={
            "brand": "NonExistentBrand",
            "series": "Some-Series",
        })
        assert resp.status_code == 404

    def test_predict_missing_brand(self, client):
        resp = client.post("/predict", json={
            "series": "Apple-iPhone-15",
        })
        assert resp.status_code == 422


class TestBatchPredict:
    def test_batch_predict_valid(self, client):
        resp = client.post("/batch_predict", json={
            "items": [
                {"brand": "Apple", "series": "Apple-iPhone-15", "storage_gb": 128},
                {"brand": "Samsung", "series": "Samsung-Galaxy-S24", "storage_gb": 256},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert pred["predicted_price_jd"] > 0

    def test_batch_predict_empty_list(self, client):
        resp = client.post("/batch_predict", json={"items": []})
        assert resp.status_code == 422  # min_length=1


class TestDeal:
    def test_deal_score_good(self, client):
        resp = client.post("/deal?asking_price_jd=100", json={
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "storage_gb": 128,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "deal_score_pct" in data
        assert "deal_label" in data

    def test_deal_score_zero_asking_price(self, client):
        resp = client.post("/deal?asking_price_jd=0", json={
            "brand": "Apple",
            "series": "Apple-iPhone-15",
        })
        assert resp.status_code == 422


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/models/info")
        if metrics is None:
            assert resp.status_code == 503
        else:
            assert resp.status_code == 200
            data = resp.json()
            assert "model_name" in data
            assert "mae" in data
            assert "r2" in data
