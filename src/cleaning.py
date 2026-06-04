import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import re
from typing import Optional

from src.paths import RAW_DIR, CLEANED_DATA_PATH
from src.config import get as cfg
from src.log import get as get_log

log = get_log("cleaning")


def normalize_arabic(text: str) -> str:
    text = str(text).replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ى', 'ي').replace('ة', 'ه')
    return text.lower()


def extract_series(title: str, brand: str) -> str:
    """Extract device series from title."""
    title = normalize_arabic(str(title))
    brand = str(brand).lower()

    # Normalize variations
    title = re.sub(r'ا?ي?ف?ون|آ?ی?ف?ون', 'iphone', title)
    title = re.sub(r'\bipone\b', 'iphone', title)
    title = re.sub(r'\bهاتف\b', 'iphone', title)
    title = re.sub(r'\bi\s*p(?:hone)?\b', 'iphone', title)
    title = re.sub(r'اي\s*فون', 'iphone', title)
    title = re.sub(r'س?ام?س?ونج?', 'samsung', title)

    # ========== APPLE ==========
    if any(b in brand for b in ['apple', 'أبل', 'ابل']):
        # iPads FIRST — before number matching can steal "10", "12", etc.
        if re.search(r'ipad|ايباد|aybad|aypad|ipod', title):
            return "Apple-iPad"
        # Apple Watch — before number matching
        if re.search(r'watch|واتش|ساع[هة]|سيريس|watsh', title):
            return "Apple-Watch"

        # Remove "swap/trade" context and everything after it
        clean_title = re.split(r'محول|محور|للبدل|بدل\s|للبيع\s|مقابل', title)[0]

        # Pattern 1: iphone followed by a number (6-17)
        m = re.search(r'iphone\s*(\d{1,2})', clean_title)
        if m:
            val = int(m.group(1))
            if 6 <= val <= 17:
                return f"Apple-iPhone-{val}"

        # Pattern 2: bare number + modifier (13pro, 17 pro max, 16e, etc.)
        m = re.search(r'\b(1[0-7])\s*(?:pro|max|mini|plus|e|برو|ماكس|ميني|بلس)', clean_title)
        if m:
            return f"Apple-iPhone-{m.group(1)}"

        # Pattern 3: iphone + XS/XR/X/SE
        m = re.search(r'iphone\s*(xs?r?|se)', clean_title, re.IGNORECASE)
        if m:
            val_upper = m.group(1).upper().replace(' ', '')
            return f"Apple-iPhone-{val_upper}"

        # Pattern 4: Arabic اكس اس/ار (XS/XR)
        m = re.search(r'اكس\s*(اس|ار)', clean_title)
        if m:
            arabic_model_map = {'اس': 'XS', 'ار': 'XR'}
            return f"Apple-iPhone-{arabic_model_map[m.group(1)]}"

        # Pattern 5: standalone Latin xr/xs
        m = re.search(r'(?<![a-z])(x\s*[rs])(?![a-z])', clean_title)
        if m:
            return f"Apple-iPhone-{m.group(1).replace(' ', '').upper()}"

        # Pattern 6: bare iPhone number at word boundary — but NOT if it's a year
        # (2017, 2018 etc) or a count (117 مره, 17 مره)
        m = re.search(r'(?<!\d)(1[0-7])(?!\s*مر[هة])(?!\d)', clean_title)
        if m:
            val = int(m.group(1))
            if 6 <= val <= 17:
                return f"Apple-iPhone-{val}"

        # Pattern 7: Arabic number words
        arabic_num_map = {'واحد':'1','اثنان':'2','اتنين':'2','ثلاثه':'3','ثلاث':'3','اربعه':'4','اربع':'4','خمسه':'5','خمس':'5','سته':'6','سبعه':'7','سبع':'7','سفن':'7','ثمانيه':'8','ثمان':'8','تسعه':'9','تسع':'9','عشره':'10','عشر':'10'}
        for word, num in arabic_num_map.items():
            if re.search(rf'\b{word}\b', clean_title):
                return f"Apple-iPhone-{num}"

        return "Apple-Other"

    # ========== SAMSUNG ==========
    if any(b in brand for b in ['samsung', 'سامسونج']):
        if re.search(r'tab|تاب|لوحي|لوح', title):
            return "Samsung-Tab"
        if re.search(r'watch|واتش|ساع[هة]', title):
            return "Samsung-Watch"
        # Normalize Arabic "جالاكسي" to "galaxy"
        title_samsung = re.sub(r'جالاكسي|جالكسي', 'galaxy', title)
        m = re.search(r'(?:galaxy\s*)?s\s*(\d{2})', title_samsung)
        if m:
            return f"Samsung-Galaxy-S{m.group(1)}"
        m = re.search(r'(?:galaxy\s*)?a\s*(\d{2})', title_samsung)
        if m:
            return f"Samsung-Galaxy-A{m.group(1)}"
        m = re.search(r'(?:galaxy\s*)?m\s*(\d+)', title_samsung)
        if m:
            return f"Samsung-Galaxy-M{m.group(1)}"
        if re.search(r'fold|flip', title_samsung):
            return "Samsung-Z"
        if re.search(r'note', title_samsung):
            return "Samsung-Note"
        return "Samsung-Other"

    # ========== OTHER BRANDS ==========
    brand_map = {
        'xiaomi': 'Xiaomi', 'شاومي': 'Xiaomi', 'ريدمي': 'Xiaomi', 'بوكو': 'Xiaomi', 'poco': 'Xiaomi',
        'honor': 'Honor', 'هونور': 'Honor',
        'huawei': 'Huawei', 'هواوي': 'Huawei',
        'oppo': 'OPPO', 'اوبو': 'OPPO',
        'tecno': 'Tecno', 'تكنو': 'Tecno',
        'infinix': 'Infinix', 'انفينيكس': 'Infinix', 'انفينكس': 'Infinix',
        'nokia': 'Nokia', 'نوكيا': 'Nokia',
        'motorola': 'Motorola', 'موتورولا': 'Motorola',
        'realme': 'Realme', 'ريلمي': 'Realme',
        'vivo': 'Vivo', 'فيفو': 'Vivo',
        'nothing': 'Nothing', 'نوثينج': 'Nothing',
        'google': 'Google', 'جوجل': 'Google',
        'oneplus': 'OnePlus', 'ون بلس': 'OnePlus',
    }
    for k, v in brand_map.items():
        if k in brand:
            return f"{v}-Other"

    return "Other"


def normalize_condition(condition: str) -> str:
    if not isinstance(condition, str):
        return 'Unknown'
    c = str(condition).strip()
    c = normalize_arabic(c)

    if re.search(r'\bجديد\b', c) and 'مستعمل' not in c:
        return 'جديد'
    if re.search(r'ممتاز|وكالة|وكاله|شبه جديد|حالة\s*ممتاز[ةه]?', c):
        return 'مستعمل - ممتاز'
    if re.search(r'جيد|نظيف|حالة\s*جيد[ةه]?', c):
        return 'مستعمل - جيد'
    if re.search(r'سي[ئء]|مكسور|خراب|حالة\s*سي[ئء][ةه]?', c):
        return 'مستعمل - سيئ'
    if c in ['جديد', 'مستعمل - ممتاز', 'مستعمل - جيد', 'مستعمل - سيئ']:
        return c
    return 'Unknown'


def extract_storage_gb(storage_str: str, title: str) -> Optional[int]:
    def parse(text: str) -> Optional[int]:
        if not isinstance(text, str) or str(text).strip().lower() == 'nan':
            return None
        t = str(text).strip().lower()

        m = re.search(r'(\d+)\s*(?:tb|t\b|تيرا|تيرابايت)', t)
        if m:
            val = int(m.group(1)) * 1024
            if 4 <= val <= 2048:
                return val

        m = re.search(r'(\d+)\s*(?:gb|g\b|جيجا|جيجابايت)', t)
        if m:
            val = int(m.group(1))
            if 4 <= val <= 2048:
                return val

        for kw in ['ذاكرة', 'سعة', 'storage', 'rom', 'رام', 'ram']:
            m = re.search(rf'{kw}\s*(\d{{2,4}})\s*(?:gb|g)?', t)
            if m:
                val = int(m.group(1))
                if val in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
                    return val

        for val in [2048, 1024, 512, 256, 128, 64, 32, 16, 8]:
            if re.search(rf'\b{val}\b', t):
                return val

        return None

    val = parse(str(storage_str))
    if val is not None:
        return val
    return parse(title)


# ── Filter stages ──────────────────────────────────────────────

def filter_delivery_service(df: pd.DataFrame) -> pd.DataFrame:
    if 'flex' in df.columns:
        mask = df['flex'] == 'خدمة التوصيل'
        log.info(f"Removing {mask.sum()} non-phone rows")
        df = df[~mask]
    return df


def filter_telecom(df: pd.DataFrame) -> pd.DataFrame:
    TELECOM = cfg('cleaning.telecom_operators', ['زين', 'اورانج', 'امنية'])
    if 'darkgraycolor2' in df.columns:
        mask = df['darkgraycolor2'].isin(TELECOM)
        log.info(f"Removing {mask.sum()} telecom rows")
        df = df[~mask]
    return df


def filter_fake_brands(df: pd.DataFrame) -> pd.DataFrame:
    COND_AS_BRAND = cfg('cleaning.condition_as_brand', ['جديد', 'مستعمل'])
    if 'darkgraycolor2' in df.columns:
        mask = df['darkgraycolor2'].isin(COND_AS_BRAND)
        log.info(f"Removing {mask.sum()} fake brand rows")
        df = df[~mask]
    return df


def filter_watches_and_tablets(df: pd.DataFrame) -> pd.DataFrame:
    title_col = 'Title' if 'Title' in df.columns else 'title'
    if title_col not in df.columns:
        return df

    watch_kw = r'watch|واتش|ساعة|ساعه|ساعات|tab|تابلت|تاب|لوحي|لوح|whatch|band|باند|سواره|سوارة|سوار|سيريس'
    mask = df[title_col].str.contains(watch_kw, case=False, na=False)
    mask_norm = df[title_col].apply(
        lambda t: bool(re.search(watch_kw, normalize_arabic(str(t)), re.IGNORECASE)) if pd.notna(t) else False
    )
    mask = mask | mask_norm

    brand_watch_kw = r'(?:huawei|هواوي|amazfit|اميزفيت).{0,15}(?:fit\s*\d|gt\s*\d|bip\s*\d)'
    brand_watch_mask = df[title_col].str.contains(brand_watch_kw, case=False, na=False)
    brand_watch_norm = df[title_col].apply(
        lambda t: bool(re.search(brand_watch_kw, normalize_arabic(str(t)), re.IGNORECASE)) if pd.notna(t) else False
    )
    mask = mask | brand_watch_mask | brand_watch_norm
    log.info(f"Removing {mask.sum()} watch/tablet rows")
    return df[~mask]


def filter_accessories(df: pd.DataFrame) -> pd.DataFrame:
    title_col = 'Title' if 'Title' in df.columns else 'title'
    if title_col not in df.columns:
        return df

    acc_kw = r'ايربودز|airpod|earpod|earphone|سماعه|سماعة|شاحن|charger|كفر|جراب|حامي|قلم|pencil|بنل|سلك|كابل|cable|سكريه|screen protector|حافظة'
    mask_acc = df[title_col].str.contains(acc_kw, case=False, na=False)
    mask_acc_norm = df[title_col].apply(
        lambda t: bool(re.search(acc_kw, normalize_arabic(str(t)), re.IGNORECASE)) if pd.notna(t) else False
    )
    mask_acc = mask_acc | mask_acc_norm
    log.info(f"Removing {mask_acc.sum()} accessory rows")
    return df[~mask_acc]


def filter_installment_listings(df: pd.DataFrame) -> pd.DataFrame:
    inst_kw = ['أقساط', 'اقساط', 'دفعة', 'دفعه', 'شهري', 'قسط']
    mask = df['title'].str.contains('|'.join(inst_kw), na=False, case=False)
    return df[~mask]


def filter_price_range(df: pd.DataFrame, min_price: float = 40, max_price: float = 1500) -> pd.DataFrame:
    return df[(df['price_jd'] >= min_price) & (df['price_jd'] <= max_price)]


# ── Brand handling ─────────────────────────────────────────────

def _std_brand(b) -> str:
    if pd.isna(b):
        return "Other"
    b = str(b).lower().strip()
    b = normalize_arabic(b)
    patterns = [
        (r'ابل|apple', 'Apple'), (r'سامسونج|samsung', 'Samsung'),
        (r'شاومي|xiaomi|ريدمي', 'Xiaomi'), (r'هونور|honor', 'Honor'),
        (r'هواوي|huawei', 'Huawei'), (r'اوبو|oppo', 'OPPO'),
        (r'تكنو|tecno', 'Tecno'), (r'انفينيكس|انفينكس|infinix', 'Infinix'),
        (r'ريلمي|realme', 'Realme'), (r'فيفو|vivo', 'Vivo'),
        (r'نوكيا|nokia', 'Nokia'), (r'موتورولا|motorola', 'Motorola'),
        (r'نوثينج|نوثينغ|nothing', 'Nothing'), (r'ون\s*بلس|oneplus', 'OnePlus'),
        (r'جوجل|google', 'Google'), (r'اخري', 'Other'),
        (r'زي\s*تي\s*اي|zte', 'ZTE'), (r'سوني|sony', 'Sony'),
        (r'ايتل|itel', 'Itel'), (r'الكاتيل|alcatel', 'Alcatel'),
        (r'لينوفو|lenovo', 'Lenovo'), (r'اسوس|asus', 'Asus'),
        (r'تي\s*سي\s*ال|tcl', 'TCL'), (r'اتش\s*تي\s*سي|htc', 'HTC'),
        (r'بلاك\s*بيري|blackberry', 'BlackBerry'), (r'بلاك\s*فيو|blackview', 'Blackview'),
        (r'ديل|dell', 'Dell'), (r'ال\s*جي|lg', 'LG'),
        (r'امازون|amazon', 'Amazon'), (r'اميزفيت|amazfit', 'Amazfit'),
        (r'جارمن|garmin', 'Garmin'), (r'دوجي|doogee', 'Doogee'),
        (r'مايكروسوفت|microsoft', 'Microsoft'), (r'موديو|modio', 'Modio'),
        (r'اوكيتيل|oukitel', 'Oukitel'), (r'فيرتو|vertu', 'Vertu'),
        (r'فيكوشا', 'Vikusha'), (r'ل8\s*ستار|l8\s*star', 'L8Star'),
        (r'ايليفون', 'Elephone'), (r'اوتيتو', 'Other'),
    ]
    for pattern, name in patterns:
        if re.search(pattern, b):
            return name
    return b.capitalize()


def _recover_brand(title: str) -> str:
    t = normalize_arabic(str(title))
    if re.search(r'iphone|ايفون|apple|ابل', t):
        return 'Apple'
    if re.search(r'samsung|سامسونج|جالاكسي', t):
        return 'Samsung'
    if re.search(r'xiaomi|شاومي|ريدمي|بوكوفون', t):
        return 'Xiaomi'
    if re.search(r'huawei|هواوي', t):
        return 'Huawei'
    if re.search(r'honor|هونور', t):
        return 'Honor'
    if re.search(r'oppo|اوبو', t):
        return 'OPPO'
    if re.search(r'tecno|تكنو', t):
        return 'Tecno'
    if re.search(r'infinix|انفينيكس|انفينكس', t):
        return 'Infinix'
    return 'Other'


# ── Main pipeline ──────────────────────────────────────────────

def clean_data(raw_path: Path, output_path: Path) -> pd.DataFrame:
    log.info(f"Reading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, encoding='utf-8-sig')
    log.info(f"Initial row count: {len(df)}")

    # Filtering
    df = filter_delivery_service(df)
    df = filter_telecom(df)
    df = filter_fake_brands(df)
    df = filter_watches_and_tablets(df)
    df = filter_accessories(df)
    log.info(f"Row count after filtering: {len(df)}")

    # Column mapping
    col_map = {
        'Title': 'title',
        'redcolor': 'price_raw',
        'darkgraycolor2': 'brand',
        'flex3': 'storage_raw',
        'borderbottomdashed': 'color_actual',
        'flex': 'model_name',
        'borderbottomdashed4': 'condition'
    }
    df = df.rename(columns=col_map)
    for col in ['title', 'price_raw', 'brand', 'storage_raw', 'color_actual', 'model_name', 'condition']:
        if col not in df.columns:
            df[col] = np.nan

    # Clean model_name noise
    model_name_noise = ['اخرى', 'أخرى', 'غير محدد', 'جديد', 'مستعمل', 'خدمة التوصيل',
                        'مستعمل - حالة ممتازة', 'مستعمل - حالة جيدة', 'مستعمل - حالة سيئة',
                        'غيرمحدد']
    if 'model_name' in df.columns:
        df.loc[df['model_name'].isin(model_name_noise), 'model_name'] = np.nan

    # Fix brand bleed
    bleed_mask = df['brand'].isin(['جديد', 'مستعمل'])
    df.loc[bleed_mask, 'condition'] = df.loc[bleed_mask, 'brand']
    df.loc[bleed_mask, 'brand'] = df.loc[bleed_mask, 'title'].apply(_recover_brand)

    # Standardize brands
    df['brand'] = df['brand'].apply(_std_brand)
    brand_counts = df['brand'].value_counts()
    rare_brands = brand_counts[brand_counts < cfg('cleaning.rare_brand_threshold', 5)].index
    df['brand'] = df['brand'].apply(lambda b: 'Other' if b in rare_brands else b)

    # Feature extraction
    df['price_jd'] = df['price_raw'].apply(
        lambda x: float(re.sub(r'[^\d]', '', str(x))) if pd.notna(x) and re.sub(r'[^\d]', '', str(x)) else np.nan
    )
    df['storage_gb'] = df.apply(lambda r: extract_storage_gb(str(r['storage_raw']), str(r['title'])), axis=1)
    df['condition'] = df['condition'].apply(normalize_condition)
    df['series'] = df.apply(lambda r: extract_series(r['title'], r['brand']), axis=1)

    series_counts = df['series'].value_counts()
    rare_series = series_counts[series_counts < cfg('cleaning.rare_series_threshold', 5)].index
    df['series'] = df['series'].apply(lambda s: 'Other-Other' if s in rare_series else s)

    # Phone age in months (depreciation feature)
    from src.features import _PHONE_RELEASE_YEARS
    import datetime

    def compute_age_months(series_str):
        s = str(series_str).lower()
        for model_key, release_year in _PHONE_RELEASE_YEARS.items():
            if model_key in s:
                return max(0, (datetime.datetime.now().year - release_year) * 12)
        return np.nan

    df['phone_age_months'] = df['series'].apply(compute_age_months)
    df['phone_age_months'] = df.groupby(['brand'])['phone_age_months'].transform(
        lambda x: x.fillna(x.median()) if not pd.isna(x.median()) else x
    )
    df['phone_age_months'] = df['phone_age_months'].fillna(df['phone_age_months'].median())

    # Final filters
    df = filter_installment_listings(df)
    df = filter_price_range(df, cfg('cleaning.min_price_jd', 40), cfg('cleaning.max_price_jd', 1500))
    df = df.dropna(subset=['price_jd', 'brand'])

    # Impute missing storage
    df['storage_gb'] = df.groupby(['brand', 'series'])['storage_gb'].transform(
        lambda x: x.fillna(x.median()) if not pd.isna(x.median()) else x
    )
    df['storage_gb'] = df.groupby(['brand'])['storage_gb'].transform(
        lambda x: x.fillna(x.median()) if not pd.isna(x.median()) else x
    )
    df['storage_gb'] = df['storage_gb'].fillna(df['storage_gb'].median())

    final_cols = ['title', 'price_jd', 'brand', 'model_name', 'storage_gb', 'color_actual', 'condition', 'series', 'phone_age_months']
    df_new = df[final_cols]

    # Save (merge with existing)
    if output_path.exists():
        log.info(f"Merging with existing data at {output_path}...")
        df_old = pd.read_csv(output_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        initial = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['title', 'price_jd'], keep='first')
        log.info(f"Removed {initial - len(df_combined)} duplicates.")
        df_final = df_combined
    else:
        df_final = df_new

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    log.info(f"Cleaned data saved. Total records: {len(df_final)}")
    return df_final


if __name__ == "__main__":
    csv_files = list(RAW_DIR.glob("*.csv"))
    raw_path = csv_files[0] if csv_files else RAW_DIR / "raw.csv"
    clean_data(raw_path, CLEANED_DATA_PATH)
