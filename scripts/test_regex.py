import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cleaning import extract_series

test_titles = [
    ("13Pro Max 512GB", "Apple"),
    ("iPhone 17 Pro Max", "Apple"),
    ("ايفون 16 برو ماكس", "Apple"),
    ("IPhone 14 Pro 256g", "Apple"),
    ("I phone 13 pro max", "Apple"),
    ("13 pro max 256g", "Apple"),
    ("iPhone XR", "Apple"),
    ("ايباد 128GB", "Apple"),
    ("Apple Watch Series 8", "Apple"),
]

for title, brand in test_titles:
    result = extract_series(title, brand)
    print(f"{title[:40]:40} -> {result}")
