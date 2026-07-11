"""
FastAPI serving endpoint for Heart Disease risk prediction.
Exposes /predict, /health, /predict-batch, /model-info and /metrics (Prometheus).
"""
import logging
import sys
import time
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import CATEGORICAL_FEATURES, engineer_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Risk API",
    description=(
        "Predicts heart disease risk from patient health data. "
        "Uses a Random Forest classifier trained on the UCI Heart Disease (Cleveland) dataset. "
        "Features include clinical measurements and derived biomarkers (heart rate reserve, "
        "BP category per JNC-8, cholesterol risk tier per ATP III)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Liveness and readiness checks"},
        {"name": "predict", "description": "Model inference endpoints"},
        {"name": "model", "description": "Model metadata and runtime stats"},
    ],
)

Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    "heart_risk_predictions_total",
    "Total predictions by risk level",
    ["risk_level"],
)
prediction_latency = Histogram(
    "heart_risk_prediction_duration_seconds",
    "Time taken per prediction",
)
batch_size_histogram = Histogram(
    "heart_risk_batch_size",
    "Batch prediction sizes",
    buckets=[1, 5, 10, 25, 50, 100],
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

# Track runtime stats in memory for /stats endpoint
_stats = {"total_predictions": 0, "high_risk": 0, "low_risk": 0}

RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


class PatientData(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 3
    }}}

    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="1=male, 0=female")
    cp: int = Field(..., description="Chest pain: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic")  # noqa: E501
    trestbps: float = Field(..., description="Resting blood pressure (mmHg)")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: int = Field(..., description="Resting ECG: 0=normal, 1=ST-T wave, 2=LV hypertrophy")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of peak ST: 1=upsloping, 2=flat, 3=downsloping")
    ca: float = Field(..., description="Number of major vessels coloured by fluoroscopy (0-3)")
    thal: float = Field(..., description="Thal: 3=normal, 6=fixed defect, 7=reversable defect")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk: str
    heart_rate_reserve: float
    age_thalach_ratio: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    count: int
    high_risk_count: int
    low_risk_count: int


def _make_prediction(data: PatientData) -> PredictionResponse:
    df = pd.DataFrame([data.model_dump()])
    df = engineer_features(df)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(float)

    start = time.time()
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])
    prediction_latency.observe(time.time() - start)

    risk = "high" if prediction == 1 else "low"
    prediction_counter.labels(risk_level=risk).inc()
    _stats["total_predictions"] += 1
    _stats[f"{risk}_risk"] += 1

    logger.info(
        "PREDICT age=%.0f sex=%d cp=%d prob=%.4f result=%s hr_reserve=%.1f",
        data.age, data.sex, data.cp, probability, risk,
        float(df["heart_rate_reserve"].iloc[0]),
    )

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk=risk,
        heart_rate_reserve=round(float(df["heart_rate_reserve"].iloc[0]), 2),
        age_thalach_ratio=round(float(df["age_thalach_ratio"].iloc[0]), 4),
    )


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model": "heart-disease-classifier", "version": "1.0.0"}


@app.get("/ready", tags=["health"])
def ready():
    if pipeline is None:
        return {"ready": False, "reason": "model not loaded"}
    return {"ready": True}


@app.get("/model-info", tags=["model"])
def model_info():
    clf = pipeline.named_steps["classifier"]
    return {
        "model_type": type(clf).__name__,
        "n_features_in": int(pipeline.named_steps["preprocessor"].n_features_in_),
        "engineered_features": [
            "heart_rate_reserve", "age_thalach_ratio",
            "st_slope_interaction", "bp_category", "chol_risk"
        ],
        "training_dataset": "UCI Heart Disease — Cleveland (297 samples)",
        "target": "Binary: 0=No Disease, 1=Disease Present",
    }


@app.get("/stats", tags=["model"])
def stats():
    total = _stats["total_predictions"]
    high = _stats["high_risk"]
    low = _stats["low_risk"]
    return {
        "total_predictions": total,
        "high_risk": high,
        "low_risk": low,
        "high_risk_rate": round(high / total, 4) if total > 0 else 0.0,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(data: PatientData):
    return _make_prediction(data)


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["predict"])
def predict_batch(patients: List[PatientData]):
    if len(patients) == 0:
        raise HTTPException(status_code=400, detail="Batch must contain at least one patient.")
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 patients.")

    batch_size_histogram.observe(len(patients))
    results = [_make_prediction(p) for p in patients]
    high = sum(1 for r in results if r.risk == "high")
    return BatchPredictionResponse(
        results=results,
        count=len(results),
        high_risk_count=high,
        low_risk_count=len(results) - high,
    )
