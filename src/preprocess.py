"""
src/preprocess.py
──────────────────
Defines the reusable feature engineering and preprocessing pipeline for the
Heart Disease UCI dataset.

Key design decisions:
  - All transformations live in a single sklearn ColumnTransformer so that
    the same preprocessing is applied at both training time and inference time.
  - The transformer is wrapped in a Pipeline with the classifier so that the
    entire fitted pipeline (scaler + encoder + model) is saved as one object.
  - engineer_features() adds 5 clinically motivated derived features BEFORE
    the sklearn pipeline runs — these are deterministic formulas, not fitted
    transformers, so they are safe to call at any time.

Usage:
    from preprocess import load_data, build_pipeline, engineer_features, ALL_FEATURES, TARGET
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ── Feature group definitions ─────────────────────────────────────────────────
# These lists are the single source of truth for which features belong to
# which transformation group. api/main.py imports CATEGORICAL_FEATURES to
# ensure inference uses the same cast logic as training.

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
    # Derived clinical features (added by engineer_features below)
    "heart_rate_reserve", "age_thalach_ratio", "st_slope_interaction"
]

CATEGORICAL_FEATURES = [
    "cp", "restecg", "slope", "thal",   # raw UCI categorical features
    "bp_category", "chol_risk"           # derived ordinal features
]

BINARY_FEATURES = [
    "sex", "fbs", "exang"  # already 0/1 — no transformation needed
]

TARGET = "target"  # binary label: 0=no disease, 1=disease present

# Combined list used for selecting model input columns
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Raw UCI features (before feature engineering) — used by predict.py for input validation
RAW_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
    "cp", "restecg", "slope", "thal", "sex", "fbs", "exang"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 5 clinically motivated derived features to the dataframe.

    Each feature is grounded in established medical guidelines:
      - heart_rate_reserve  : exercise physiology (max achievable HR minus actual)
      - age_thalach_ratio   : age-adjusted cardiac fitness
      - st_slope_interaction: AHA ST-segment guidelines
      - bp_category         : JNC-8 hypertension classification (0-3 ordinal)
      - chol_risk           : NCEP ATP III cholesterol risk tiers (0-2 ordinal)

    This function is called BEFORE the sklearn pipeline — it runs the same
    deterministic formulas at training time and at inference time.
    """
    df = df.copy()  # never modify the caller's dataframe in place

    # Heart Rate Reserve: (predicted max HR based on age) minus (achieved max HR)
    # Formula: 220 - age gives the age-predicted maximum heart rate.
    # A higher reserve means the heart didn't work hard during the test,
    # which can indicate impaired cardiac output in disease patients.
    df["heart_rate_reserve"] = (220 - df["age"]) - df["thalach"]

    # Age-normalised heart rate: divides achieved HR by age.
    # A 150 bpm result means very different things for a 30-year-old vs a 70-year-old.
    # This ratio captures fitness relative to age-adjusted expected capacity.
    df["age_thalach_ratio"] = df["thalach"] / df["age"]

    # ST Slope Interaction: multiplies ST depression magnitude by slope direction.
    # slope: 1=upsloping, 2=flat, 3=downsloping. Downsloping + high oldpeak
    # is the highest-risk ST pattern according to AHA exercise testing guidelines.
    df["st_slope_interaction"] = df["oldpeak"] * df["slope"]

    # Blood pressure category per JNC-8 guidelines (ordinal 0-3):
    # 0=Normal (<120 mmHg), 1=Elevated (120-129), 2=Stage1 HTN (130-139), 3=Stage2 HTN (≥140)
    def bp_cat(sbp):
        if sbp < 120:
            return 0
        elif sbp < 130:
            return 1
        elif sbp < 140:
            return 2
        else:
            return 3

    df["bp_category"] = df["trestbps"].apply(bp_cat)

    # Cholesterol risk tier per NCEP ATP III guidelines (ordinal 0-2):
    # 0=Desirable (<200 mg/dl), 1=Borderline High (200-239), 2=High (≥240)
    def chol_cat(chol):
        if chol < 200:
            return 0
        elif chol < 240:
            return 1
        else:
            return 2

    df["chol_risk"] = df["chol"].apply(chol_cat)

    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Load the cleaned CSV, drop rows with missing values, and run feature engineering.

    Note: 6 rows have NaN in 'ca' and 'thal' — they are dropped here (2% of data).
    Imputation was not used because it would introduce artificial patterns into
    two already important features at such a small scale.
    """
    df = pd.read_csv(path)
    df = df.dropna()              # drop the 6 rows with missing ca/thal values
    df = engineer_features(df)    # add the 5 derived clinical features
    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer that handles all feature transformations.

    Three transformation groups:
      - Numeric  : StandardScaler → zero mean, unit variance
                   Required for Logistic Regression (scale-sensitive).
      - Categorical: OneHotEncoder(drop='first') → binary columns per category
                   drop='first' avoids the dummy variable trap (multicollinearity).
      - Binary   : passthrough → sex, fbs, exang are already 0/1
    """
    return ColumnTransformer(
        transformers=[
            # StandardScaler: subtract mean, divide by std dev → features on same scale
            ("num", StandardScaler(), NUMERIC_FEATURES),

            # OneHotEncoder: nominal categories → binary columns
            # sparse_output=False returns a dense array (easier to work with)
            # drop='first' removes one column per feature to avoid multicollinearity
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),

            # passthrough: keep binary features as-is, no transformation needed
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    """
    Wrap preprocessor + classifier into a single sklearn Pipeline.

    Saving this pipeline as one object (joblib.dump) means:
    - The fitted scaler parameters (mean, std) are preserved
    - The fitted encoder categories are preserved
    - At inference time, a single pipeline.predict(X) call applies
      all transformations and returns predictions — no separate
      preprocessing step needed
    """
    return Pipeline([
        ("preprocessor", get_preprocessor()),  # step 1: transform features
        ("classifier",   classifier),           # step 2: predict
    ])
