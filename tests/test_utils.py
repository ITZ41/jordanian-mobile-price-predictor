import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import extract_price, extract_storage, clean_text


class TestExtractPrice:
    def test_basic(self):
        assert extract_price("629 دينار") == 629.0

    def test_with_comma(self):
        assert extract_price("1,200 دينار") == 1200.0

    def test_empty(self):
        assert extract_price("") is None

    def test_none(self):
        assert extract_price(None) is None

    def test_bare_number(self):
        assert extract_price("500") == 500.0


class TestExtractStorage:
    def test_gb(self):
        assert extract_storage("256 جيجابايت") == 256

    def test_tb(self):
        assert extract_storage("1 TB") == 1024

    def test_bare_number(self):
        assert extract_storage("128") == 128

    def test_empty(self):
        assert extract_storage("") is None

    def test_none(self):
        assert extract_storage(None) is None


class TestCleanText:
    def test_extra_spaces(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_empty(self):
        assert clean_text("") == ""

    def test_none(self):
        assert clean_text(None) == ""
