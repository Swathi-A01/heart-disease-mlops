import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Fixtures come from conftest.py: api_client, sample_high_risk, sample_low_risk


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()


def test_ready(api_client):
    r = api_client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_predict_returns_all_fields(api_client, sample_low_risk):
    r = api_client.post("/predict", json=sample_low_risk)
    assert r.status_code == 200
    body = r.json()
    for field in ("prediction", "probability", "risk", "heart_rate_reserve", "age_thalach_ratio"):
        assert field in body, f"Missing field: {field}"


def test_predict_binary_output(api_client, sample_low_risk):
    r = api_client.post("/predict", json=sample_low_risk)
    assert r.json()["prediction"] in (0, 1)


def test_predict_probability_range(api_client, sample_high_risk, sample_low_risk):
    for patient in [sample_low_risk, sample_high_risk]:
        r = api_client.post("/predict", json=patient)
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"


def test_predict_risk_label_consistency(api_client, sample_high_risk, sample_low_risk):
    for patient in [sample_low_risk, sample_high_risk]:
        r = api_client.post("/predict", json=patient)
        body = r.json()
        assert body["risk"] in ("low", "high")
        if body["prediction"] == 1:
            assert body["risk"] == "high"
        else:
            assert body["risk"] == "low"


def test_predict_invalid_input_422(api_client):
    r = api_client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_missing_single_field(api_client, sample_low_risk):
    incomplete = {k: v for k, v in sample_low_risk.items() if k != "thalach"}
    r = api_client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_model_info_endpoint(api_client):
    r = api_client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "model_type" in body
    assert "engineered_features" in body
    assert len(body["engineered_features"]) >= 3


def test_stats_endpoint(api_client, sample_low_risk):
    api_client.post("/predict", json=sample_low_risk)
    r = api_client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 1
    assert "high_risk_rate" in body


def test_predict_batch_valid(api_client, sample_high_risk, sample_low_risk):
    r = api_client.post("/predict-batch", json=[sample_low_risk, sample_high_risk])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["high_risk_count"] + body["low_risk_count"] == 2
    assert len(body["results"]) == 2


def test_predict_batch_empty_raises(api_client):
    r = api_client.post("/predict-batch", json=[])
    assert r.status_code == 400


def test_predict_batch_over_limit(api_client, sample_low_risk):
    r = api_client.post("/predict-batch", json=[sample_low_risk] * 101)
    assert r.status_code == 400


def test_heart_rate_reserve_present(api_client, sample_low_risk):
    r = api_client.post("/predict", json=sample_low_risk)
    assert "heart_rate_reserve" in r.json()


def test_response_time_under_500ms(api_client, sample_low_risk):
    start = time.time()
    r = api_client.post("/predict", json=sample_low_risk)
    elapsed_ms = (time.time() - start) * 1000
    assert r.status_code == 200
    assert elapsed_ms < 500, f"Response took {elapsed_ms:.0f}ms — too slow"
