# Jordanian Mobile Price Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![CI](https://github.com/ITZ41/jordanian-mobile-price-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ITZ41/jordanian-mobile-price-predictor/actions/workflows/ci.yml)

> **Predict second-hand mobile phone prices in the Jordanian market using machine learning — trained on real Arabic marketplace listings.**

A production-grade ML pipeline that scrapes Arabic listings from Jordanian marketplaces (OpenSooq/السوق المفتوح), cleans messy unstructured Arabic text, engineers features, trains and compares gradient boosting models, and serves predictions through a bilingual (English/Arabic) Streamlit dashboard and FastAPI REST API.

**GitHub Topics:** `machine-learning` `nlp` `arabic-nlp` `gradient-boosting` `streamlit` `fastapi` `price-prediction` `scikit-learn` `xgboost` `docker` `jordan` `open-sooq`

---

## Results

| Metric | Value |
|---|---|
| **MAE** | ~57 JOD |
| **R²** | 0.82 |
| **Training Samples** | ~2,800 listings |
| **Model** | XGBoost (auto-selected over HistGradientBoosting) |
| **Cross-Validation** | 5-fold |

### Sample Predictions

| Phone | Condition | Predicted | Actual |
|---|---|---|---|
| iPhone 15 Pro Max 256GB | مستعمل - ممتاز | ~850 JOD | 850 JOD |
| Galaxy S24 Ultra 512GB | جديد | ~780 JOD | 780 JOD |
| iPhone 13 128GB | مستعمل - ممتاز | ~480 JOD | 480 JOD |
| Redmi Note 13 256GB | مستعمل - ممتاز | ~195 JOD | 195 JOD |



## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌───────────┐  │
│  │  Raw CSV  │───▶│ Cleaning  │───▶│  Features  │───▶│  Model    │  │
│  │ (Arabic)  │    │ (Arabic   │    │ (Target    │    │ Training  │  │
│  │           │    │  NLP)     │    │  Encoding) │    │ (3 models)│  │
│  └──────────┘    └───────────┘    └────────────┘    └─────┬─────┘  │
│                                                            │        │
│                                                            ▼        │
│                                                      ┌───────────┐  │
│                                                      │  Best     │  │
│                                                      │  Model    │  │
│                                                      │ (joblib)  │  │
│                                                      └─────┬─────┘  │
│                                                            │        │
└────────────────────────────────────────────────────────────┼────────┘
                                                             │
                    ┌────────────────────────────────────────┼────┐
                    │              SERVING                   │    │
                    │                                        ▼    │
                    │  ┌──────────────┐         ┌──────────────┐  │
                    │  │  Streamlit   │         │   FastAPI    │  │
                    │  │  Dashboard   │         │   REST API   │  │
                    │  │  (port 8501) │         │  (port 8000) │  │
                    │  │              │         │              │  │
                    │  │ • Price Est  │         │ • /predict   │  │
                    │  │ • Market     │         │ • /deal      │  │
                    │  │ • Trade-In   │         │ • /batch     │  │
                    │  │ • Compare    │         │ • /models    │  │
                    │  │ • Analyzer   │         │              │  │
                    │  └──────────────┘         └──────────────┘  │
                    │                                             │
                    │         Bilingual: English ⟷ Arabic         │
                    └─────────────────────────────────────────────┘
```

### Stage Details

1. **Data Cleaning** (`src/cleaning.py`) — Arabic text normalization (أ/إ/آ → ا, ى → ي, ة → ه), brand/series extraction from messy titles using regex patterns, condition normalization, swap/trade context detection (بدل/محول), price filtering, rare brand/series removal
2. **Feature Engineering** (`src/features.py`) — Title-based binary features (Pro/Max/Ultra/5G/eSIM/warranty/sealed), target encoding for categoricals (brand, series, condition), phone age from release year, storage and RAM extraction
3. **Model Training** (`src/train_model.py`) — Trains 3 models (HistGradientBoosting, XGBoost, LightGBM) with 5-fold CV, auto-selects winner by MAE, fine-tunes with grid search, generates SHAP feature importance and learning curves
4. **Validation** (`src/validate.py`) — Schema validation on cleaned data, data quality report with nulls/outliers/rare categories
5. **Serving** — Streamlit dashboard (5 tabs, dark theme, bilingual) + FastAPI REST API (predict, deal score, batch predict, model info)

---

## Design Decisions

### Why HistGradientBoosting vs XGBoost vs LightGBM?
All three are gradient boosting implementations. HistGradientBoosting (scikit-learn) handles missing values natively and requires no extra dependency. XGBoost is the industry standard for tabular data. LightGBM is often faster on large datasets. We train all three with cross-validation and let the data pick the winner — no assumptions.

### Why Target Encoding over One-Hot?
Brand and series have high cardinality (50+ series). One-hot encoding would create sparse columns and risk overfitting. Target encoding replaces each category with the mean price of that category, which captures the price signal in a single numeric column. We use scikit-learn's `TargetEncoder` with smoothing to prevent overfitting on rare categories.

### Why FastAPI over Flask?
FastAPI gives us automatic OpenAPI docs, Pydantic validation, async support, and type-safe request/response models out of the box. For a prediction API, this means self-documenting endpoints and proper error handling with minimal code.

### Why log-transform the target?
Phone prices are right-skewed (many cheap phones, few expensive ones). Predicting `log(1 + price)` normalizes the distribution, which improves gradient boosting performance. We exponentiate back to JOD for display.

### Why Arabic NLP without a library?
Off-the-shelf Arabic NLP tools (CAMeL, Farasa) are heavy and designed for MSA (Modern Standard Arabic). Jordanian marketplace titles are dialectal, mixed with English, and full of abbreviations. Custom regex patterns are lighter, faster, and more accurate for this specific domain.

---

## Project Structure

```
├── app/app.py                 # Streamlit dashboard (5 tabs, bilingual)
├── api/serve.py               # FastAPI prediction API
├── src/
│   ├── cleaning.py            # Data cleaning & Arabic NLP
│   ├── features.py            # Feature engineering pipeline
│   ├── train_model.py         # Model training, selection & tuning
│   ├── validate.py            # Data schema validation & quality report
│   ├── analysis.py            # Market analysis charts (matplotlib/seaborn)
│   ├── generate_report.py     # Text summary report
│   ├── config.py              # YAML config loader (dot notation)
│   ├── paths.py               # Shared path constants
│   ├── log.py                 # Structured logging
│   └── utils.py               # Misc utilities
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_cleaning.py       # 36+ tests for Arabic NLP
│   ├── test_features.py       # Feature extraction tests
│   ├── test_model.py          # Model loading & prediction tests
│   ├── test_api.py            # API endpoint tests
│   └── test_utils.py          # Utility function tests
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── scripts/                   # Debugging/inspection helpers
├── config.yaml                # Centralized configuration
├── models/                    # Trained model + metrics (gitignored)
├── data/
│   ├── raw/                   # Raw scrapes (gitignored)
│   └── processed/             # cleaned_data.csv + sample_data.csv
├── reports/                   # Generated figures & reports
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Clean Raw Data

```bash
PYTHONPATH=. python src/cleaning.py
```

Or use the included sample data to skip this step:
```bash
cp data/processed/sample_data.csv data/processed/cleaned_data.csv
```

### 3. Validate Data

```bash
PYTHONPATH=. python src/validate.py
```

### 4. Train the Model

```bash
PYTHONPATH=. python src/train_model.py
```

This trains 3 models (HistGradientBoosting, XGBoost, LightGBM) via 5-fold CV, picks the winner, fine-tunes it, and saves:
- `models/price_model.joblib` — trained model
- `models/metrics.json` — MAE, R²
- `models/feature_importance.json` — SHAP-based feature importance
- `models/experiment_log.jsonl` — full experiment history
- `reports/learning_curve.png` — learning curve plot
- `reports/shap_importance.png` — SHAP beeswarm plot

### 5. Launch the Dashboard

```bash
streamlit run app/app.py
```

### 6. Start the API

```bash
PYTHONPATH=. python api/serve.py
```

### Docker (Both Services)

```bash
docker compose up --build
# Dashboard → http://localhost:8501
# API       → http://localhost:8000
# API docs  → http://localhost:8000/docs
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/brands` | GET | List all brands |
| `/brands/{brand}/series` | GET | List series for a brand |
| `/predict` | POST | Price prediction (single) |
| `/deal` | POST | Deal score evaluation |
| `/batch_predict` | POST | Batch price prediction (list) |
| `/models/info` | GET | Model metadata (type, date, MAE, samples) |

### Example: Predict Price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"brand":"Apple","series":"Apple-iPhone-15","storage_gb":128,"condition":"مستعمل - ممتاز"}'
```

Response:
```json
{
  "predicted_price_jd": 720.00,
  "lower_bound_jd": 663.00,
  "upper_bound_jd": 777.00,
  "model_mae": 57.0,
  "model_name": "XGBoost"
}
```

### Example: Batch Predict

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[{"brand":"Apple","series":"Apple-iPhone-15","storage_gb":128},
        {"brand":"Samsung","series":"Samsung-Galaxy-S24","storage_gb":256}]'
```

---

## Dashboard Tabs

1. ** Price Estimate** — Predict a phone's market value with confidence interval, deal score, and comparable listings. Use "All" storage for full market breakdown.
2. ** Market Analysis** — Interactive Plotly charts: price distribution, series breakdown, condition analysis.
3. ** Trade-In Calculator** — Estimate upgrade options when trading in your current phone.
4. ** Brand Comparison** — Side-by-side value retention comparison across brands.
5. ** My Listing Analyzer** — Paste a raw Arabic listing title and get automatic brand/series/storage/condition extraction + predicted price + deal score.

---

## Testing

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

Coverage target: >60%. Tests cover:
- Arabic NLP: series extraction (iPhone 17, iPad priority, Watch detection, swap context, count exclusion, Samsung Arabic)
- Feature engineering: TitleFeatureExtractor output shapes and values
- Model: loaded model accepts valid input and returns positive float
- API: all endpoints with valid and invalid inputs
- Utilities: price/storage extraction, text cleaning

---

## Configuration

All tunable parameters live in `config.yaml`:

- `cleaning.*` — Price bounds, rare-brand/series thresholds, delivery keywords
- `model.*` — Hyperparameters for all 3 models and the tuning grid
- `app.*` — Dashboard port and default MAE fallback
- `api.*` — API host and port

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | scikit-learn (HistGBM), XGBoost, LightGBM |
| Explainability | SHAP |
| ML Pipeline | scikit-learn Pipeline + joblib |
| Frontend | Streamlit |
| Charts | Plotly, matplotlib, seaborn |
| API | FastAPI + Uvicorn |
| Config | YAML (PyYAML) |
| Testing | pytest + pytest-cov |
| Linting | ruff |
| CI/CD | GitHub Actions |
| Deployment | Docker + docker-compose |

---

## License

MIT — see [LICENSE](LICENSE).
