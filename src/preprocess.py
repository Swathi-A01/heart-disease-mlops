"""
Reusable preprocessing pipeline for the Heart Disease UCI dataset.
Fitted on training data only — call transform() at inference time.

Also exposes engineer_features() which adds clinically meaningful derived
features before the sklearn pipeline runs. These are domain-specific choices:
- heart_rate_reserve: difference between predicted max HR and achieved max HR.
  Lower reserve → worse cardiac fitness → higher disease risk.
- bp_category: JNC-8 guideline blood pressure stages (0-3 ordinal).
- chol_risk: ATP III guideline cholesterol risk tiers (0-2 ordinal).
- age_thalach_ratio: normalises max HR by age — captures fitness relative to age.
- st_slope_interaction: combines oldpeak and slope as a joint ST-segment signal.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca",
                    "heart_rate_reserve", "age_thalach_ratio", "st_slope_interaction"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal", "bp_category", "chol_risk"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
TARGET = "target"

RAW_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca",
                "cp", "restecg", "slope", "thal", "sex", "fbs", "exang"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Heart rate reserve: predicted max HR (220 - age) minus achieved max HR.
    # A large positive reserve means the heart didn't work hard → possible impairment.
    df["heart_rate_reserve"] = (220 - df["age"]) - df["thalach"]

    # Age-normalised heart rate: thalach / age — captures fitness relative to age.
    df["age_thalach_ratio"] = df["thalach"] / df["age"]

    # ST-segment combined signal: product of ST depression and slope severity.
    # Slope is coded 1=up, 2=flat, 3=down — downsloping + high depression is high risk.
    df["st_slope_interaction"] = df["oldpeak"] * df["slope"]

    # Blood pressure category per JNC-8 guidelines (ordinal: 0-3).
    def bp_cat(sbp):
        if sbp < 120:
            return 0   # Normal
        elif sbp < 130:
            return 1   # Elevated
        elif sbp < 140:
            return 2   # Stage 1 hypertension
        else:
            return 3   # Stage 2 hypertension
    df["bp_category"] = df["trestbps"].apply(bp_cat)

    # Cholesterol risk tier per ATP III guidelines (ordinal: 0-2).
    def chol_cat(chol):
        if chol < 200:
            return 0   # Desirable
        elif chol < 240:
            return 1   # Borderline high
        else:
            return 2   # High
    df["chol_risk"] = df["chol"].apply(chol_cat)

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    df = engineer_features(df)
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
