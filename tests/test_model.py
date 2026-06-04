"""Tests for model loading and prediction behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np


class TestModelLoading:
    """Test that the model loads and has expected attributes."""

    def test_model_is_pipeline(self, loaded_model):
        from sklearn.pipeline import Pipeline
        assert isinstance(loaded_model, Pipeline)

    def test_model_has_preprocessor(self, loaded_model):
        assert "prep" in loaded_model.named_steps

    def test_model_has_regressor(self, loaded_model):
        assert "reg" in loaded_model.named_steps


class TestModelPrediction:
    """Test model prediction on valid inputs."""

    def test_predict_returns_positive_float(self, loaded_model, sample_input_df):
        prediction = np.expm1(loaded_model.predict(sample_input_df)[0])
        assert np.isfinite(prediction)
        assert prediction > 0

    def test_predict_apple_iphone(self, loaded_model):
        import pandas as pd
        input_df = pd.DataFrame([{
            "title": "Apple Apple-iPhone-15 256GB مستعمل - ممتاز",
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "storage_gb": 256,
            "phone_age_months": 12,
            "condition": "مستعمل - ممتاز",
            "model_name": "Apple Apple-iPhone-15",
            "color_actual": "Unknown",
        }])
        prediction = np.expm1(loaded_model.predict(input_df)[0])
        assert 100 < prediction < 2000  # reasonable range for iPhone 15

    def test_predict_samsung_galaxy(self, loaded_model):
        import pandas as pd
        input_df = pd.DataFrame([{
            "title": "Samsung Samsung-Galaxy-S24 256GB مستعمل - ممتاز",
            "brand": "Samsung",
            "series": "Samsung-Galaxy-S24",
            "storage_gb": 256,
            "phone_age_months": 6,
            "condition": "مستعمل - ممتاز",
            "model_name": "Samsung Samsung-Galaxy-S24",
            "color_actual": "Unknown",
        }])
        prediction = np.expm1(loaded_model.predict(input_df)[0])
        assert 50 < prediction < 2000

    def test_higher_storage_higher_price(self, loaded_model):
        """Higher storage should generally yield higher price for same model."""
        import pandas as pd
        base = {
            "title": "Apple Apple-iPhone-15 128GB مستعمل - ممتاز",
            "brand": "Apple",
            "series": "Apple-iPhone-15",
            "phone_age_months": 12,
            "condition": "مستعمل - ممتاز",
            "model_name": "Apple Apple-iPhone-15",
            "color_actual": "Unknown",
        }
        df_low = pd.DataFrame([{**base, "storage_gb": 128}])
        df_high = pd.DataFrame([{**base, "storage_gb": 512}])
        pred_low = np.expm1(loaded_model.predict(df_low)[0])
        pred_high = np.expm1(loaded_model.predict(df_high)[0])
        assert pred_high >= pred_low * 0.8  # allow some tolerance
