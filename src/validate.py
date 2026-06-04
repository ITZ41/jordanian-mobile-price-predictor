"""Data validation module — schema checks and quality report for cleaned data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.paths import CLEANED_DATA_PATH
from src.config import get as cfg
from src.log import get as get_log

log = get_log("validate")


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Validate the schema of the cleaned dataframe. Returns list of error messages."""
    errors: list[str] = []

    # Required columns
    required = ["title", "price_jd", "brand", "series", "storage_gb", "condition"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors  # Can't check further without required columns

    # price_jd: must be numeric within bounds
    price_min = cfg("validation.price_min", 10)
    price_max = cfg("validation.price_max", 5000)
    if df["price_jd"].dtype not in [np.float64, np.int64, np.float32, np.int32]:
        errors.append(f"price_jd has invalid dtype: {df['price_jd'].dtype}")
    bad_prices = df[(df["price_jd"] < price_min) | (df["price_jd"] > price_max)]
    if len(bad_prices) > 0:
        errors.append(f"{len(bad_prices)} rows with price outside [{price_min}, {price_max}]")

    # brand: must be non-null string
    null_brands = df["brand"].isnull().sum()
    if null_brands > 0:
        errors.append(f"{null_brands} rows with null brand")

    # series: must be non-null string
    null_series = df["series"].isnull().sum()
    if null_series > 0:
        errors.append(f"{null_series} rows with null series")

    # condition: must be one of the valid set
    valid_conditions = cfg(
        "validation.valid_conditions",
        ["جديد", "مستعمل - ممتاز", "مستعمل - جيد", "مستعمل - سيئ", "Unknown"],
    )
    invalid_cond = df[~df["condition"].isin(valid_conditions)]["condition"].unique()
    if len(invalid_cond) > 0:
        errors.append(f"Invalid conditions found: {list(invalid_cond)}")

    # storage_gb: must be positive if present
    storage_min = cfg("validation.storage_min", 1)
    storage_max = cfg("validation.storage_max", 2048)
    if df["storage_gb"].notna().any():
        bad_storage = df[
            (df["storage_gb"].notna())
            & ((df["storage_gb"] < storage_min) | (df["storage_gb"] > storage_max))
        ]
        if len(bad_storage) > 0:
            errors.append(f"{len(bad_storage)} rows with storage outside [{storage_min}, {storage_max}]")

    return errors


def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality report. Returns dict of metrics."""
    report: dict = {}

    report["total_rows"] = len(df)
    report["columns"] = list(df.columns)

    # Nulls per column
    nulls = df.isnull().sum()
    report["nulls_per_column"] = {col: int(nulls[col]) for col in df.columns if nulls[col] > 0}

    # Price statistics
    report["price_mean"] = round(float(df["price_jd"].mean()), 2)
    report["price_median"] = round(float(df["price_jd"].median()), 2)
    report["price_std"] = round(float(df["price_jd"].std()), 2)
    report["price_min"] = float(df["price_jd"].min())
    report["price_max"] = float(df["price_jd"].max())

    # Price outliers (beyond 3 std from mean)
    mean = df["price_jd"].mean()
    std = df["price_jd"].std()
    outliers = df[(df["price_jd"] < mean - 3 * std) | (df["price_jd"] > mean + 3 * std)]
    report["price_outliers_count"] = len(outliers)

    # Brand distribution
    brand_counts = df["brand"].value_counts()
    report["unique_brands"] = len(brand_counts)
    report["top_5_brands"] = brand_counts.head(5).to_dict()
    rare_brands = brand_counts[brand_counts < 5]
    report["rare_brands_count"] = len(rare_brands)

    # Series distribution
    series_counts = df["series"].value_counts()
    report["unique_series"] = len(series_counts)
    rare_series = series_counts[series_counts < 5]
    report["rare_series_count"] = len(rare_series)

    # Condition distribution
    report["condition_distribution"] = df["condition"].value_counts().to_dict()

    # Storage stats
    if "storage_gb" in df.columns:
        report["storage_mean"] = round(float(df["storage_gb"].mean()), 1)
        report["storage_nulls"] = int(df["storage_gb"].isnull().sum())

    return report


def print_report(report: dict) -> None:
    """Pretty-print the data quality report."""
    print("\n" + "=" * 60)
    print("       DATA QUALITY REPORT")
    print("=" * 60)
    print(f"\nTotal rows: {report['total_rows']}")
    print(f"Columns: {', '.join(report['columns'])}")

    print(f"\n--- Price ---")
    print(f"  Mean:   {report['price_mean']} JOD")
    print(f"  Median: {report['price_median']} JOD")
    print(f"  Std:    {report['price_std']} JOD")
    print(f"  Range:  {report['price_min']} – {report['price_max']} JOD")
    print(f"  Outliers (>3σ): {report['price_outliers_count']}")

    print(f"\n--- Brands ---")
    print(f"  Unique: {report['unique_brands']}")
    print(f"  Rare (<5 listings): {report['rare_brands_count']}")
    print(f"  Top 5: {report['top_5_brands']}")

    print(f"\n--- Series ---")
    print(f"  Unique: {report['unique_series']}")
    print(f"  Rare (<5 listings): {report['rare_series_count']}")

    print(f"\n--- Condition ---")
    for cond, cnt in report.get("condition_distribution", {}).items():
        print(f"  {cond}: {cnt}")

    if "storage_mean" in report:
        print(f"\n--- Storage ---")
        print(f"  Mean: {report['storage_mean']} GB")
        print(f"  Nulls: {report['storage_nulls']}")

    if report.get("nulls_per_column"):
        print(f"\n--- Nulls ---")
        for col, cnt in report["nulls_per_column"].items():
            print(f"  {col}: {cnt}")

    print("\n" + "=" * 60)


def validate_and_report(csv_path: Path = CLEANED_DATA_PATH) -> bool:
    """Run full validation and print report. Returns True if data passes all checks."""
    if not csv_path.exists():
        log.error(f"Data not found at {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    log.info(f"Validating {len(df)} rows from {csv_path}")

    # Schema validation
    errors = validate_schema(df)
    if errors:
        log.error("Schema validation FAILED:")
        for err in errors:
            log.error(f"  ✗ {err}")
        return False

    log.info("Schema validation passed ✓")

    # Quality report
    report = data_quality_report(df)
    print_report(report)
    log.info("Data quality report generated")

    return True


if __name__ == "__main__":
    success = validate_and_report()
    if not success:
        sys.exit(1)
