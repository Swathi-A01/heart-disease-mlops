"""
tests/conftest.py
──────────────────
Shared pytest fixtures — automatically available to all test files.
pytest loads this file before running any tests.

Why fixtures?
  Without fixtures, each test file would independently load the dataset and
  model, which is slow (model loading alone takes ~0.5s). Using session-scoped
  fixtures means the heavy setup happens once per test session, not once per test.

Fixture scopes:
  scope="session"  — created once when the test session starts, shared by all tests
  scope="function" — created fresh for each individual test (default)

Usage in test files:
  def test_something(df):           # df fixture injected automatically
  def test_api(api_client, sample_high_risk):  # multiple fixtures combined
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src/ to Python path so test files can import preprocess, etc.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_PATH  = Path(__file__).parent.parent / "data" / "heart.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "pipeline.pkl"


@pytest.fixture(scope="session")
def raw_df():
    """
    Raw CSV dataframe — no feature engineering applied.
    Used in tests that verify the raw data properties (missing values, target encoding).
    Session-scoped: loaded once and reused across all tests that need it.
    """
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available — run data/download_data.py first")
    return pd.read_csv(str(DATA_PATH)).dropna()


@pytest.fixture(scope="session")
def df():
    """
    Fully preprocessed dataframe including all 5 engineered features.
    Used in tests that verify the feature engineering pipeline.
    Session-scoped: load_data() runs once and the result is reused.
    """
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available — run data/download_data.py first")
    from preprocess import load_data
    return load_data(str(DATA_PATH))


@pytest.fixture(scope="session")
def trained_pipeline():
    """
    Loaded sklearn Pipeline (preprocessor + classifier).
    Session-scoped: joblib.load() runs once — avoids repeated disk reads.
    Skips if the model hasn't been trained yet (run src/train.py first).
    """
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained — run src/train.py first")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def api_client(trained_pipeline):
    """
    FastAPI TestClient — session-scoped for speed.
    TestClient wraps the ASGI app so HTTP calls don't require a running server.
    The trained_pipeline fixture dependency ensures the model is loaded before
    the client is created (avoids import-time failure if model is missing).
    """
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_high_risk():
    """
    Test patient with multiple known heart disease risk factors:
    - Age 67, male
    - Asymptomatic chest pain (cp=4) — highest-risk type
    - Very low max HR (108) — indicates poor cardiac capacity
    - 3 blocked vessels (ca=3) — direct disease indicator
    - Reversable thal defect (thal=7)
    Expected: prediction=1, probability ~0.998
    """
    return {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
        "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7,
    }


@pytest.fixture
def sample_low_risk():
    """
    Test patient with good cardiac profile:
    - Age 35, female
    - Typical angina (cp=1) — least serious type
    - High max HR (180) — excellent cardiac capacity for age
    - No exercise angina, no blocked vessels
    - Normal thal (thal=3)
    Expected: prediction=0, probability ~0.04
    """
    return {
        "age": 35, "sex": 0, "cp": 1, "trestbps": 105, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
    }
