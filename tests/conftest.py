"""Shared test fixtures for the Jordanian Mobile Price Predictor test suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
import joblib

from src.paths import MODEL_PATH, CLEANED_DATA_PATH


@pytest.fixture
def sample_cleaned_df():
    """Return a small sample cleaned dataframe for testing."""
    return pd.DataFrame([
        {
            "title": "ايفون 15 برو ماكس 256 مستعمل ممتاز",
            "price_jd": 850.0,
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "storage_gb": 256,
            "condition": "مستعمل - ممتاز",
            "model_name": "Apple iPhone 15",
            "color_actual": "Unknown",
            "phone_age_months": 18,
        },
        {
            "title": "سامسونج جالاكسي S24 الترا 512 جديد",
            "price_jd": 780.0,
            "brand": "Samsung",
            "series": "Samsung-Galaxy-S24",
            "storage_gb": 512,
            "condition": "جديد",
            "model_name": "Samsung Galaxy S24",
            "color_actual": "Unknown",
            "phone_age_months": 6,
        },
        {
            "title": "ايفون 14 برو 128 مستعمل جيد",
            "price_jd": 620.0,
            "brand": "Apple",
            "series": "Apple-iPhone-14",
            "storage_gb": 128,
            "condition": "مستعمل - جيد",
            "model_name": "Apple iPhone 14",
            "color_actual": "Unknown",
            "phone_age_months": 30,
        },
        {
            "title": "شاومي ريدمي نوت 13 256 مستعمل ممتاز",
            "price_jd": 195.0,
            "brand": "Xiaomi",
            "series": "Xiaomi-Other",
            "storage_gb": 256,
            "condition": "مستعمل - ممتاز",
            "model_name": "Xiaomi Redmi",
            "color_actual": "Unknown",
            "phone_age_months": 12,
        },
        {
            "title": "سامسونج جالاكسي A54 128 مستعمل جيد",
            "price_jd": 165.0,
            "brand": "Samsung",
            "series": "Samsung-Galaxy-A54",
            "storage_gb": 128,
            "condition": "مستعمل - جيد",
            "model_name": "Samsung Galaxy A54",
            "color_actual": "Unknown",
            "phone_age_months": 24,
        },
    ])


@pytest.fixture
def loaded_model():
    """Load the trained model, or skip test if not available."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_input_df():
    """Return a single-row DataFrame formatted for model prediction."""
    return pd.DataFrame([{
        "title": "Apple Apple-iPhone-15 128GB مستعمل - ممتاز",
        "brand": "Apple",
        "series": "Apple-iPhone-15",
        "storage_gb": 128,
        "phone_age_months": 18,
        "condition": "مستعمل - ممتاز",
        "model_name": "Apple Apple-iPhone-15",
        "color_actual": "Unknown",
    }])
