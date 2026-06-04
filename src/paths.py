from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = RAW_DIR  # use glob to find actual file
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
MODEL_PATH = MODELS_DIR / "price_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
REPORT_PATH = REPORTS_DIR / "summary_report.txt"
FIGURES_DIR = REPORTS_DIR / "figures"
