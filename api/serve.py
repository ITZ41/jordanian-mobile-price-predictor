import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.paths import MODEL_PATH, METRICS_PATH, CLEANED_DATA_PATH
from src.config import get as cfg

app = FastAPI(
    title="Mobile Price Engine API",
    description="Predict second-hand mobile phone prices in the Jordanian market",
    version="1.0.0",
)

model = None
metrics = None
df = None


@app.on_event("startup")
def startup():
    global model, metrics, df
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH)
        for col in ['brand', 'series']:
            df[col] = df[col].fillna("Other").astype(str)


class PredictRequest(BaseModel):
    brand: str
    series: str
    storage_gb: int = 128
    phone_age_months: float = 0
    condition: str = "مستعمل - ممتاز"
    is_pro: bool = False
    is_max: bool = False
    is_ultra: bool = False
    is_plus: bool = False
    has_warranty: bool = False
    is_sealed: bool = False


class PredictResponse(BaseModel):
    predicted_price_jd: float
    lower_bound_jd: float
    upper_bound_jd: float
    model_mae: float
    model_name: str


class DealResponse(PredictResponse):
    asking_price_jd: float
    deal_score_pct: float
    deal_label: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "data_loaded": df is not None}


@app.get("/brands")
def list_brands():
    if df is None:
        raise HTTPException(503, "Data not loaded")
    brands = sorted(df['brand'].unique().tolist())
    return {"brands": brands}


@app.get("/brands/{brand}/series")
def list_series(brand: str):
    if df is None:
        raise HTTPException(503, "Data not loaded")
    brand_df = df[df['brand'] == brand]
    counts = brand_df['series'].value_counts()
    available = sorted(counts[counts >= 5].index.tolist())
    return {"brand": brand, "series": available}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run src/train_model.py first.")

    title_parts = [req.brand, req.series, f"{req.storage_gb}GB", req.condition]
    if req.is_pro:
        title_parts.append("pro")
    if req.is_max:
        title_parts.append("max")
    if req.is_ultra:
        title_parts.append("ultra")
    if req.is_plus:
        title_parts.append("plus")
    if req.has_warranty:
        title_parts.append("كفالة")
    if req.is_sealed:
        title_parts.append("كرتون مسكر")

    input_df = pd.DataFrame([{
        'title': " ".join(title_parts),
        'brand': req.brand,
        'series': req.series,
        'storage_gb': req.storage_gb,
        'phone_age_months': req.phone_age_months,
        'condition': req.condition,
        'model_name': f"{req.brand} {req.series}",
        'color_actual': 'Unknown',
    }])

    try:
        prediction = float(np.expm1(model.predict(input_df)[0]))
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

    mae = metrics.get("mae", cfg("app.default_mae", 57.39)) if metrics else cfg("app.default_mae", 57.39)
    model_name = metrics.get("model", "unknown") if metrics else "unknown"

    return PredictResponse(
        predicted_price_jd=round(prediction, 2),
        lower_bound_jd=round(prediction - mae, 2),
        upper_bound_jd=round(prediction + mae, 2),
        model_mae=mae,
        model_name=model_name,
    )


@app.post("/deal", response_model=DealResponse)
def deal_score(req: PredictRequest, asking_price_jd: float):
    pred = predict(req)
    if asking_price_jd <= 0:
        raise HTTPException(400, "asking_price_jd must be positive")

    deal_pct = (pred.predicted_price_jd - asking_price_jd) / pred.predicted_price_jd * 100

    if deal_pct > 10:
        label = "Great Deal"
    elif deal_pct > 0:
        label = "Fair Price"
    elif deal_pct > -10:
        label = "Slightly Overpriced"
    else:
        label = "Overpriced"

    return DealResponse(
        **pred.model_dump(),
        asking_price_jd=asking_price_jd,
        deal_score_pct=round(deal_pct, 1),
        deal_label=label,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg("api.host", "0.0.0.0"), port=cfg("api.port", 8000))
