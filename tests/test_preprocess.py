import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess import (  # noqa: E402
    load_data, get_preprocessor, build_pipeline,
    engineer_features, ALL_FEATURES, TARGET
)

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"


@pytest.fixture
def df():
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available — run data/download_data.py first")
    return load_data(str(DATA_PATH))


@pytest.fixture
def raw_df():
    import pandas as pd
    if not DATA_PATH.exists():
        pytest.skip("Dataset not available")
    return pd.read_csv(str(DATA_PATH)).dropna()


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


def test_engineer_features_adds_columns(raw_df):
    df_eng = engineer_features(raw_df)
    for col in ["heart_rate_reserve", "age_thalach_ratio", "st_slope_interaction",
                "bp_category", "chol_risk"]:
        assert col in df_eng.columns, f"Missing engineered feature: {col}"


def test_heart_rate_reserve_formula(raw_df):
    df_eng = engineer_features(raw_df)
    expected = (220 - raw_df["age"]) - raw_df["thalach"]
    assert np.allclose(df_eng["heart_rate_reserve"].values, expected.values)


def test_bp_category_bounds(raw_df):
    df_eng = engineer_features(raw_df)
    assert df_eng["bp_category"].between(0, 3).all()


def test_chol_risk_bounds(raw_df):
    df_eng = engineer_features(raw_df)
    assert df_eng["chol_risk"].between(0, 2).all()


def test_model_accuracy_above_threshold(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    assert acc >= 0.75, f"Model accuracy {acc:.2f} below minimum threshold of 0.75"
