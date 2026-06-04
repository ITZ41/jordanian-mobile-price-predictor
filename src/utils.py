"""
Utility functions for Jordanian Market Price Analyzer
"""
import re
import csv


def extract_price(price_str: str) -> float | None:
    """Extract numeric price from string like '629 دينار'"""
    if not price_str:
        return None
    nums = re.findall(r'[\d,]+', str(price_str))
    if nums:
        try:
            return float(nums[0].replace(',', ''))
        except ValueError:
            return None
    return None


def extract_storage(storage_str: str) -> int | None:
    """Extract storage in GB from string like '256 جيجابايت'"""
    if not storage_str:
        return None
    nums = re.findall(r'\d+', str(storage_str))
    if nums:
        try:
            val = int(nums[0])
            # Convert TB to GB
            if 'تيرا' in storage_str or 'TB' in storage_str.upper():
                val *= 1024
            return val
        except ValueError:
            return None
    return None


def clean_text(text: str) -> str:
    """Clean Arabic text: remove extra spaces, normalize"""
    if not text:
        return ''
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(filepath: str) -> list[dict]:
    """Load CSV data and return list of dicts"""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 8:
                rows.append({
                    'title': row[0],
                    'time_posted': row[1],
                    'price_raw': row[2],
                    'brand': row[3],
                    'model': row[4],
                    'storage_raw': row[5],
                    'color': row[6],
                    'condition': row[7],
                })
    return rows
