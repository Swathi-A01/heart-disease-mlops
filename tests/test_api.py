import sys
import time
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


SAMPLE_LOW_RISK = {
    "age": 45, "sex": 0, "cp": 1, "trestbps": 110,
    "chol": 190, "fbs": 0, "restecg": 0, "thalach": 170,
    "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
}

SAMPLE_HIGH_RISK = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160,
    "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108,
    "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()


def test_predict_returns_all_fields(client):
    r = client.post("/predict", json=SAMPLE_LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    for field in ("prediction", "probability", "risk", "heart_rate_reserve", "age_thalach_ratio"):
        assert field in body, f"Missing field: {field}"


def test_predict_binary_output(client):
    r = client.post("/predict", json=SAMPLE_LOW_RISK)
    assert r.json()["prediction"] in (0, 1)


def test_predict_probability_range(client):
    for patient in [SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK]:
        r = client.post("/predict", json=patient)
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"


def test_predict_risk_label_consistency(client):
    for patient in [SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK]:
        r = client.post("/predict", json=patient)
        body = r.json()
        assert body["risk"] in ("low", "high")
        if body["prediction"] == 1:
            assert body["risk"] == "high"
        else:
            assert body["risk"] == "low"


def test_predict_invalid_input_422(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_missing_single_field(client):
    incomplete = {k: v for k, v in SAMPLE_LOW_RISK.items() if k != "thalach"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_model_info_endpoint(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "model_type" in body
    assert "engineered_features" in body
    assert len(body["engineered_features"]) >= 3


def test_stats_endpoint(client):
    client.post("/predict", json=SAMPLE_LOW_RISK)
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 1
    assert "high_risk_rate" in body


def test_predict_batch_valid(client):
    r = client.post("/predict-batch", json=[SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["high_risk_count"] + body["low_risk_count"] == 2
    assert len(body["results"]) == 2


def test_predict_batch_empty_raises(client):
    r = client.post("/predict-batch", json=[])
    assert r.status_code == 400


def test_predict_batch_over_limit(client):
    big_batch = [SAMPLE_LOW_RISK] * 101
    r = client.post("/predict-batch", json=big_batch)
    assert r.status_code == 400


def test_heart_rate_reserve_is_positive_for_healthy(client):
    r = client.post("/predict", json=SAMPLE_LOW_RISK)
    # Low-risk patient has high thalach → lower heart rate reserve
    assert "heart_rate_reserve" in r.json()


def test_response_time_under_500ms(client):
    start = time.time()
    r = client.post("/predict", json=SAMPLE_LOW_RISK)
    elapsed_ms = (time.time() - start) * 1000
    assert r.status_code == 200
    assert elapsed_ms < 500, f"Response took {elapsed_ms:.0f}ms — too slow"
