import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cleaning import extract_series, normalize_condition, normalize_arabic


class TestExtractSeries:
    # iPhone number variants
    def test_iphone_number(self):
        assert extract_series("iPhone 15 Pro Max", "Apple") == "Apple-iPhone-15"

    def test_iphone_arabic(self):
        assert extract_series("ايفون 16 برو ماكس", "Apple") == "Apple-iPhone-16"

    def test_iphone_shorthand(self):
        assert extract_series("13Pro Max 512GB", "Apple") == "Apple-iPhone-13"

    def test_iphone_xr(self):
        assert extract_series("iPhone XR", "Apple") == "Apple-iPhone-XR"

    def test_iphone_xs(self):
        assert extract_series("iPhone XS Max", "Apple") == "Apple-iPhone-XS"

    def test_iphone_se(self):
        assert extract_series("iPhone SE 2022", "Apple") == "Apple-iPhone-SE"

    def test_iphone_8(self):
        assert extract_series("iPhone 8 Plus", "Apple") == "Apple-iPhone-8"

    # iPhone 17
    def test_iphone_17(self):
        assert extract_series("iPhone 17", "Apple") == "Apple-iPhone-17"

    def test_iphone_17_pro_max(self):
        assert extract_series("Iphone 17 pro max 256G", "Apple") == "Apple-iPhone-17"

    def test_iphone_17_arabic(self):
        assert extract_series("ايفون 17 برو ماكس", "Apple") == "Apple-iPhone-17"

    def test_iphone_17_bare(self):
        assert extract_series("17 pro max 256g", "Apple") == "Apple-iPhone-17"

    # iPad must be detected BEFORE number matching
    def test_ipad(self):
        assert extract_series("ايباد 128GB", "Apple") == "Apple-iPad"

    def test_ipad_pro_2017(self):
        assert extract_series("ايباد برو 2017", "Apple") == "Apple-iPad"

    def test_ipad_10th_gen(self):
        assert extract_series("ايباد 10 الجيل التاسع", "Apple") == "Apple-iPad"

    def test_ipad_12_9_2017(self):
        assert extract_series("ايباد برو 12.9 انش 2017", "Apple") == "Apple-iPad"

    def test_ipad_air(self):
        assert extract_series("iPad Air 5", "Apple") == "Apple-iPad"

    # Watch must be detected BEFORE number matching
    def test_apple_watch(self):
        assert extract_series("Apple Watch Series 8", "Apple") == "Apple-Watch"

    def test_apple_watch_arabic(self):
        assert extract_series("ساعة ابل الترا", "Apple") == "Apple-Watch"

    def test_apple_watch_arabic_norm(self):
        assert extract_series("ساعه ابل سيريس 8", "Apple") == "Apple-Watch"

    # Swap/trade context handling
    def test_swap_first_phone_wins(self):
        assert extract_series("Iphone 15 pro max محول Iphone 17 pro max", "Apple") == "Apple-iPhone-15"

    def test_swap_xs(self):
        assert extract_series("ايفون xs max محول 17", "Apple") == "Apple-iPhone-XS"

    # Numbers that are NOT phone models (counts, years)
    def test_not_phone_17_count(self):
        assert extract_series("ايفون 16e مشحون 17 مره", "Apple") == "Apple-iPhone-16"

    def test_not_phone_117_count(self):
        assert extract_series("ايفون وكاله مشحون 117 مره", "Apple") == "Apple-iPhone-11"

    # Samsung
    def test_samsung_s_series(self):
        assert extract_series("Galaxy S24 Ultra", "Samsung") == "Samsung-Galaxy-S24"

    def test_samsung_s_arabic(self):
        assert extract_series("جالاكسي S24 الترا", "Samsung") == "Samsung-Galaxy-S24"

    def test_samsung_a_series(self):
        assert extract_series("Galaxy A54", "Samsung") == "Samsung-Galaxy-A54"

    def test_samsung_m_series(self):
        assert extract_series("Galaxy M14", "Samsung") == "Samsung-Galaxy-M14"

    def test_samsung_fold(self):
        assert extract_series("Galaxy Z Fold 5", "Samsung") == "Samsung-Z"

    def test_samsung_tab(self):
        assert extract_series("Galaxy Tab S9", "Samsung") == "Samsung-Tab"

    def test_samsung_watch(self):
        assert extract_series("Galaxy Watch 6", "Samsung") == "Samsung-Watch"

    def test_samsung_a_arabic_brand(self):
        assert extract_series("سامسونج A55", "Samsung") == "Samsung-Galaxy-A55"

    # Other brands
    def test_xiaomi(self):
        assert extract_series("Redmi Note 13", "Xiaomi") == "Xiaomi-Other"

    def test_other_brand(self):
        assert extract_series("Some phone", "Other") == "Other"


class TestNormalizeCondition:
    def test_new(self):
        assert normalize_condition("جديد") == "جديد"

    def test_excellent(self):
        assert normalize_condition("مستعمل - حالة ممتازة") == "مستعمل - ممتاز"

    def test_excellent_agency(self):
        assert normalize_condition("وكالة") == "مستعمل - ممتاز"

    def test_good(self):
        assert normalize_condition("مستعمل - حالة جيدة") == "مستعمل - جيد"

    def test_bad(self):
        assert normalize_condition("مستعمل - حالة سيئة") == "مستعمل - سيئ"

    def test_bad_broken(self):
        assert normalize_condition("مكسور") == "مستعمل - سيئ"

    def test_non_string(self):
        assert normalize_condition(None) == "Unknown"
        assert normalize_condition(123) == "Unknown"

    def test_already_normalized(self):
        assert normalize_condition("مستعمل - ممتاز") == "مستعمل - ممتاز"


class TestNormalizeArabic:
    def test_aleph_variants(self):
        assert normalize_arabic("أحمد") == normalize_arabic("احمد")

    def test_taa_marbuta(self):
        assert normalize_arabic("حالة") == normalize_arabic("حاله")

    def test_lower(self):
        result = normalize_arabic("SAMSUNG")
        assert result == "samsung"
