"""
FastAPI serving endpoint for Heart Disease risk prediction.
Exposes /predict, /health, and /metrics (Prometheus).
"""
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from prometheus_client import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from patient health data.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    "heart_risk_predictions_total",
    "Total predictions by risk level",
    ["risk_level"],
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)


class PatientData(BaseModel):
    age: float = Field(..., example=55, description="Age in years")
    sex: int = Field(..., example=1, description="1=male, 0=female")
    cp: int = Field(..., example=0, description="Chest pain type (0-3)")
    trestbps: float = Field(..., example=130, description="Resting blood pressure")
    chol: float = Field(..., example=250, description="Serum cholesterol mg/dl")
    fbs: int = Field(..., example=0, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: int = Field(..., example=0, description="Resting ECG results (0-2)")
    thalach: float = Field(..., example=150, description="Max heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., example=1.5, description="ST depression induced by exercise")
    slope: int = Field(..., example=1, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., example=0, description="Number of major vessels (0-3)")
    thal: float = Field(..., example=2, description="Thal: 1=normal, 2=fixed defect, 3=reversable defect")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "heart-disease-classifier"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])
    risk = "high" if prediction == 1 else "low"

    prediction_counter.labels(risk_level=risk).inc()
    logger.info(
        "PREDICT age=%.0f sex=%d cp=%d prob=%.4f result=%s",
        data.age, data.sex, data.cp, probability, risk,
    )

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk=risk,
    )
