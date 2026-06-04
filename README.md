# Jordanian Mobile Price Predictor

A production-ready machine learning project that predicts second-hand mobile phone prices in the Jordanian market. Built with scikit-learn, XGBoost, and Streamlit.

## Features

- **Smart Data Cleaning** — Arabic text normalization, brand/series extraction from messy titles, condition normalization, swap/trade context handling
- **Dual Model Training** — Compares HistGradientBoosting vs XGBoost, auto-selects the winner by cross-validation MAE, then fine-tunes it
- **Feature Engineering** — Target encoding, title-based feature extraction (Pro/Max/ultra/warranty/5G/eSIM), phone age, and RAM
- **Interactive Dashboard** — Dark-themed Streamlit UI with price estimation, market analysis, trade-in calculator, and brand comparison
- **Bilingual UI** — Full English/Arabic toggle across all dashboard tabs
- **"All Storage" Mode** — Select brand + series without specifying storage to see full market breakdown by storage & condition
- **REST API** — FastAPI endpoint for programmatic price predictions
- **Configurable** — All thresholds, hyperparameters, and paths centralized in `config.yaml`
- **Dockerized** — One-command startup with `docker compose`

## Project Structure

```
├── app/app.py              # Streamlit dashboard
├── api/serve.py            # FastAPI prediction API
├── src/
│   ├── cleaning.py         # Data cleaning & series extraction
│   ├── features.py         # Feature engineering (TitleFeatureExtractor, Preprocessor)
│   ├── train_model.py      # Model training & selection
│   ├── analysis.py         # Market analysis charts
│   ├── generate_report.py  # Text summary report
│   ├── config.py           # YAML config loader
│   ├── paths.py            # Shared path constants
│   ├── log.py              # Structured logging
│   └── utils.py            # Misc utilities
├── tests/                  # pytest suite (36+ tests)
├── scripts/                # Debugging/inspection helpers
├── config.yaml             # Centralized configuration
├── models/                 # Trained model + metrics (gitignored, regenerated)
├── data/
│   ├── raw/                # Raw scrapes (gitignored)
│   └── processed/          # cleaned_data.csv
├── reports/                # Generated figures & text report
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

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

### 3. Train the Model

```bash
PYTHONPATH=. python src/train_model.py
```

This compares HistGradientBoosting and XGBoost via 5-fold CV, picks the winner, fine-tunes it, and saves to `models/price_model.joblib` with `models/metrics.json`.

### 4. Launch the Dashboard

```bash
streamlit run app/app.py
```

### 5. (Optional) Start the API

```bash
PYTHONPATH=. python api/serve.py
```

### Docker

```bash
docker compose up --build
# Dashboard → http://localhost:8501
# API       → http://localhost:8000
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/brands` | GET | List all brands |
| `/brands/{brand}/series` | GET | List series for a brand |
| `/predict` | POST | Price prediction (JSON body) |
| `/deal` | POST | Deal score evaluation |

### Example: Predict Price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"brand":"Apple","series":"Apple-iPhone-15","storage_gb":128,"condition":"مستعمل - ممتاز"}'
```

## Dashboard Tabs

1. **Price Estimate** — Predict a phone's market value, see deal score, and comparable listings. Use "All" storage to see the full market breakdown for any brand + series.
2. **Market Analysis** — Price trends, distribution charts, and condition breakdowns powered by Plotly.
3. **Trade-In Calculator** — Estimate upgrade options when trading in your current phone (phone field is optional).
4. **Brand Comparison** — Side-by-side value retention comparison across brands.

## Testing

```bash
python -m pytest tests/ -v
```

36+ tests covering series extraction (iPhone 17, iPad priority, Watch detection, swap context, count exclusion, Samsung Arabic), condition normalization, and Arabic normalization.

## Configuration

All tunable parameters live in `config.yaml`:

- `cleaning.*` — Price bounds, rare-brand/series thresholds, delivery keywords
- `model.*` — Hyperparameters for both models and the tuning grid
- `app.*` — Dashboard port and default MAE fallback
- `api.*` — API host and port

## License

MIT — see [LICENSE](LICENSE).
