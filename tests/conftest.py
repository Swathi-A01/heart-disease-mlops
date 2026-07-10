"""
Shared pytest fixtures — available to all test files automatically.
pytest loads this file before any tests run.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "pipeline.pkl"


@pytest.fixture(scope="session")
def raw_df():
    """Raw CSV — no feature engineering, used for data integrity tests."""
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available — run data/download_data.py first")
    return pd.read_csv(str(DATA_PATH)).dropna()


@pytest.fixture(scope="session")
def df():
    """Fully preprocessed dataframe including engineered features."""
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available — run data/download_data.py first")
    from preprocess import load_data
    return load_data(str(DATA_PATH))


@pytest.fixture(scope="session")
def trained_pipeline():
    """Loaded sklearn pipeline — session-scoped so it loads once per test run."""
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained — run src/train.py first")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def api_client(trained_pipeline):
    """FastAPI test client — session-scoped for speed."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_high_risk():
    """Known high-risk patient — elderly male with multiple red flags."""
    return {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
        "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7,
    }


@pytest.fixture
def sample_low_risk():
    """Known low-risk patient — young female, good cardiac metrics."""
    return {
        "age": 35, "sex": 0, "cp": 1, "trestbps": 105, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
    }
