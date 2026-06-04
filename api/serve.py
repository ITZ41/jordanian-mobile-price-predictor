"""FastAPI REST API for Jordanian Mobile Price Predictor.

Endpoints:
    GET  /health          — Health check
    GET  /brands          — List all brands
    GET  /brands/{brand}/series — List series for a brand
    GET  /models/info     — Model metadata
    POST /predict         — Single price prediction
    POST /batch_predict   — Batch price prediction
    POST /deal            — Deal score evaluation
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.paths import MODEL_PATH, METRICS_PATH, CLEANED_DATA_PATH
from src.config import get as cfg
from src.log import get as get_log

log = get_log("api")

app = FastAPI(
    title="Mobile Price Engine API",
    description="Predict second-hand mobile phone prices in the Jordanian market",
    version="2.0.0",
)

model = None
metrics: Optional[dict] = None
df: Optional[pd.DataFrame] = None


# ── Request logging middleware ──────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    log.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"elapsed={elapsed:.3f}s"
    )
    return response


# ── Startup ─────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    global model, metrics, df
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        log.info(f"Model loaded from {MODEL_PATH}")
    else:
        log.warning(f"Model not found at {MODEL_PATH}")

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        log.info(f"Metrics loaded: {metrics}")
    else:
        log.warning(f"Metrics not found at {METRICS_PATH}")

    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH)
        for col in ["brand", "series"]:
            df[col] = df[col].fillna("Other").astype(str)
        log.info(f"Data loaded: {len(df)} rows")
    else:
        log.warning(f"Data not found at {CLEANED_DATA_PATH}")


# ── Pydantic models ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    brand: str = Field(..., min_length=1, description="Phone brand, e.g. Apple, Samsung")
    series: str = Field(..., min_length=1, description="Phone series, e.g. Apple-iPhone-15")
    storage_gb: int = Field(default=128, ge=1, le=2048, description="Storage in GB")
    phone_age_months: float = Field(default=0, ge=0, le=240, description="Phone age in months")
    condition: str = Field(default="مستعمل - ممتاز", description="Phone condition")
    is_pro: bool = False
    is_max: bool = False
    is_ultra: bool = False
    is_plus: bool = False
    has_warranty: bool = False
    is_sealed: bool = False

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        valid = ["جديد", "مستعمل - мمتاز", "مستعمل - جيد", "مستعمل - سيئ", "Unknown"]
        if v not in valid:
            raise ValueError(f"condition must be one of {valid}, got '{v}'")
        return v


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


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int


class ModelInfoResponse(BaseModel):
    model_name: str
    mae: float
    r2: float
    training_samples: int
    training_date: str
    features: list[str]


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


# ── Prediction helper ───────────────────────────────────────────

def _run_prediction(req: PredictRequest) -> float:
    """Run model prediction and return price in JOD."""
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
        "title": " ".join(title_parts),
        "brand": req.brand,
        "series": req.series,
        "storage_gb": req.storage_gb,
        "phone_age_months": req.phone_age_months,
        "condition": req.condition,
        "model_name": f"{req.brand} {req.series}",
        "color_actual": "Unknown",
    }])

    prediction = float(np.expm1(model.predict(input_df)[0]))
    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError("Model returned invalid prediction")
    return prediction


def _build_predict_response(prediction: float) -> PredictResponse:
    mae = metrics.get("mae", cfg("app.default_mae", 57.39)) if metrics else cfg("app.default_mae", 57.39)
    model_name = metrics.get("model", "unknown") if metrics else "unknown"
    return PredictResponse(
        predicted_price_jd=round(prediction, 2),
        lower_bound_jd=round(prediction - mae, 2),
        upper_bound_jd=round(prediction + mae, 2),
        model_mae=mae,
        model_name=model_name,
    )


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/brands")
def list_brands():
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    brands = sorted(df["brand"].unique().tolist())
    return {"brands": brands, "count": len(brands)}


@app.get("/brands/{brand}/series")
def list_series(brand: str):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    brand_df = df[df["brand"] == brand]
    if brand_df.empty:
        raise HTTPException(status_code=404, detail=f"Brand '{brand}' not found. Use GET /brands to see available brands.")
    counts = brand_df["series"].value_counts()
    available = sorted(counts[counts >= 5].index.tolist())
    return {"brand": brand, "series": available, "count": len(available)}


@app.get("/models/info", response_model=ModelInfoResponse)
def model_info():
    if metrics is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    return ModelInfoResponse(
        model_name=metrics.get("model", "unknown"),
        mae=metrics.get("mae", 0),
        r2=metrics.get("r2", 0),
        training_samples=metrics.get("training_samples", 0),
        training_date=metrics.get("training_date", "unknown"),
        features=[
            "title", "brand", "series", "storage_gb",
            "phone_age_months", "condition", "model_name", "color_actual",
        ],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train_model.py first.")

    # Validate brand exists
    if df is not None and req.brand not in df["brand"].unique():
        raise HTTPException(
            status_code=404,
            detail=f"Brand '{req.brand}' not found. Available: {sorted(df['brand'].unique().tolist())}",
        )

    try:
        prediction = _run_prediction(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")

    return _build_predict_response(prediction)


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train_model.py first.")

    predictions = []
    errors = []
    for i, item in enumerate(req.items):
        try:
            prediction = _run_prediction(item)
            predictions.append(_build_predict_response(prediction))
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            log.warning(f"Batch item {i} failed: {e}")

    if not predictions:
        raise HTTPException(status_code=400, detail=f"All predictions failed. Errors: {errors}")

    if errors:
        log.warning(f"Batch completed with {len(errors)} errors out of {len(req.items)} items")

    return BatchPredictResponse(predictions=predictions, count=len(predictions))


@app.post("/deal", response_model=DealResponse)
def deal_score(req: PredictRequest, asking_price_jd: float = 0):
    if asking_price_jd <= 0:
        raise HTTPException(status_code=422, detail="asking_price_jd must be a positive number")

    pred = predict(req)

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


# ── Global exception handler ────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg("api.host", "0.0.0.0"), port=cfg("api.port", 8000))
