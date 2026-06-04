import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import TitleFeatureExtractor


class TestTitleFeatureExtractor:
    def setup_method(self):
        self.ext = TitleFeatureExtractor().fit(None)

    def test_is_pro(self):
        titles = pd.Series(["iPhone 15 Pro", "iPhone 15"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 0] == 1  # is_pro
        assert result[1, 0] == 0

    def test_is_max(self):
        titles = pd.Series(["iPhone 15 Pro Max", "iPhone 15"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 1] == 1  # is_max
        assert result[1, 1] == 0

    def test_is_ultra(self):
        titles = pd.Series(["Galaxy S24 Ultra", "Galaxy S24"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 2] == 1  # is_ultra
        assert result[1, 2] == 0

    def test_is_plus(self):
        titles = pd.Series(["iPhone 15 Plus", "iPhone 15"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 3] == 1  # is_plus
        assert result[1, 3] == 0

    def test_has_warranty(self):
        titles = pd.Series(["iPhone 15 كفالة", "iPhone 15"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 4] == 1  # has_warranty
        assert result[1, 4] == 0

    def test_is_sealed(self):
        titles = pd.Series(["iPhone 15 كرتون مسكر", "iPhone 15"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 5] == 1  # is_sealed
        assert result[1, 5] == 0

    def test_is_new(self):
        titles = pd.Series(["iPhone 15 جديد", "iPhone 15 مستعمل"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 6] == 1  # is_new
        assert result[1, 6] == 0

    def test_arabic_pro(self):
        titles = pd.Series(["ايفون 15 برو"])
        result = self.ext.transform(titles.to_frame())
        assert result[0, 0] == 1  # is_pro (Arabic برو)
