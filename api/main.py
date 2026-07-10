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

from preprocess import CATEGORICAL_FEATURES  # noqa: E402

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
    model_config = {"json_schema_extra": {"example": {
        "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 3
    }}}

    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="1=male, 0=female")
    cp: int = Field(..., description="Chest pain: 1=typical, 2=atypical, 3=non-anginal, 4=asympt")
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


@app.get("/health")
def health():
    return {"status": "ok", "model": "heart-disease-classifier"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    # Cast categorical columns to float — OHE was fitted on float values from CSV
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(float)

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
