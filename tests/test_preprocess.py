import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess import load_data, get_preprocessor, build_pipeline, ALL_FEATURES, TARGET

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"


@pytest.fixture
def df():
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available — run data/download_data.py first")
    return load_data(str(DATA_PATH))


def test_load_data_columns(df):
    assert TARGET in df.columns
    for col in ALL_FEATURES:
        assert col in df.columns


def test_load_data_no_missing(df):
    assert df.isnull().sum().sum() == 0


def test_target_is_binary(df):
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_preprocessor_output_shape(df):
    from sklearn.model_selection import train_test_split
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = get_preprocessor()
    X_transformed = pre.fit_transform(X_train)
    assert X_transformed.shape[0] == len(X_train)
    assert X_transformed.shape[1] > len(ALL_FEATURES)


def test_pipeline_fits_and_predicts(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds).issubset({0, 1})
    probs = pipe.predict_proba(X_test)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))
