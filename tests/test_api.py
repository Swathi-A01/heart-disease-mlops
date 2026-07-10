import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_PATH = Path(__file__).parent.parent / "models" / "pipeline.pkl"


@pytest.fixture
def client():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet — run src/train.py first")
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


SAMPLE_PATIENT = {
    "age": 55, "sex": 1, "cp": 0, "trestbps": 130,
    "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_input(client):
    r = client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "probability" in body
    assert "risk" in body
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk"] in ("low", "high")


def test_predict_invalid_input(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_missing_field(client):
    incomplete = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422
