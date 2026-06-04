import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.paths import CLEANED_DATA_PATH, FIGURES_DIR
from src.log import get as get_log

log = get_log("analysis")

def run_analysis(csv_path=CLEANED_DATA_PATH, output_dir=FIGURES_DIR):
    """Generates market analysis charts."""
    if not csv_path.exists():
        log.error(f"Data not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Brand Distribution
    plt.figure(figsize=(12, 6))
    brand_counts = df['brand'].value_counts().head(10)
    sns.barplot(x=brand_counts.index, y=brand_counts.values)
    plt.title('Top 10 Brands on OpenSooq (Jordan)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'brand_distribution.png')
    plt.close()

    # 2. Price Distribution by Brand (Top 5)
    plt.figure(figsize=(12, 6))
    top_5_brands = df['brand'].value_counts().head(5).index
    df_top_5 = df[df['brand'].isin(top_5_brands)]
    sns.boxplot(x='brand', y='price_jd', data=df_top_5)
    plt.title('Price Distribution for Top 5 Brands')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'price_by_brand.png')
    plt.close()

    # 3. Condition vs Price
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='condition', y='price_jd', data=df)
    plt.title('Price by Condition')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'price_by_condition.png')
    plt.close()

    log.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    run_analysis()
