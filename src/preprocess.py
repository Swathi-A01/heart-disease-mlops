"""
Reusable preprocessing pipeline for the Heart Disease UCI dataset.
Fitted on training data only — call transform() at inference time.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
TARGET = "target"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def get_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier),
    ])
