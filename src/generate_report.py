import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.paths import CLEANED_DATA_PATH, REPORT_PATH
from src.log import get as get_log

log = get_log("generate_report")

def generate_report(csv_path=CLEANED_DATA_PATH, output_path=REPORT_PATH):
    """Generates a text summary report of the market data."""
    if not csv_path.exists():
        log.error(f"Data not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Jordanian Market Price Analyzer - Data Summary Report\n")
        f.write("====================================================\n\n")
        f.write(f"Total Listings: {len(df)}\n")
        f.write(f"Average Price: {df['price_jd'].mean():.2f} JD\n")
        f.write(f"Median Price: {df['price_jd'].median():.2f} JD\n")
        f.write(f"Price Range: {df['price_jd'].min()} - {df['price_jd'].max()} JD\n\n")

        f.write("Top 5 Brands by Volume:\n")
        f.write(df['brand'].value_counts().head(5).to_string())
        f.write("\n\n")

        f.write("Average Price by Brand (Top 10 Volume Brands):\n")
        top_brands = df['brand'].value_counts().head(10).index
        f.write(df[df['brand'].isin(top_brands)].groupby('brand')['price_jd'].mean().sort_values(ascending=False).to_string())
        f.write("\n\n")

        f.write("Condition Distribution:\n")
        f.write(df['condition'].value_counts().to_string())

    log.info(f"Report generated at {output_path}")

if __name__ == "__main__":
    generate_report()
