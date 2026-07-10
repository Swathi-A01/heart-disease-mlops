# Step 03 — Feature Engineering, Model Training & MLflow Tracking

**Date:** 2026-07-10
**Assignment Tasks:** Task 2 [8 marks] + Task 3 [5 marks] + Task 4 [7 marks]
**Status:** Complete

---

## What Was Done

### Preprocessing Pipeline (`src/preprocess.py`)

Built a `sklearn.compose.ColumnTransformer` that handles all three feature types:

| Feature Group | Features | Transformer | Why |
|--------------|----------|-------------|-----|
| Numeric | age, trestbps, chol, thalach, oldpeak, ca | `StandardScaler` | Models sensitive to scale (LR) need normalisation |
| Categorical | cp, restecg, slope, thal | `OneHotEncoder(drop='first')` | Nominal categories — no ordinal assumption |
| Binary | sex, fbs, exang | passthrough | Already 0/1 — no transformation needed |

The transformer is wrapped in `build_pipeline(classifier)` so the **exact same preprocessing
runs at both train time and inference time**. No leakage possible — the scaler is fitted only
on `X_train`.

### Models Trained (`src/train.py`)

#### Logistic Regression
- Baseline: `C=1.0` (default regularisation)
- Tuning: `GridSearchCV` over `C = [0.01, 0.1, 1, 10, 100]`, 5-fold StratifiedKFold, scoring=roc_auc
- Best params found: `C=1.0`
- **Result: ROC-AUC = 0.9531 — best performing model**

#### Random Forest
- Tuning: `RandomizedSearchCV`, 10 iterations, 5-fold CV
- Search space: `n_estimators=[100,200,300]`, `max_depth=[None,5,10,15]`, `min_samples_split=[2,5,10]`
- **Result: ROC-AUC = 0.9342**

#### XGBoost
- Tuning: `RandomizedSearchCV`, 8 iterations, 5-fold CV
- Search space: `n_estimators=[100,200]`, `max_depth=[3,5,7]`, `learning_rate=[0.01,0.1,0.2]`
- **Result: ROC-AUC = 0.8906**

### Full Results Table

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|----|
| **Logistic Regression** | **0.9531** | **0.8500** | 0.8800 | 0.7857 | 0.8302 |
| Random Forest | 0.9342 | 0.8333 | 0.8462 | 0.7857 | 0.8148 |
| XGBoost | 0.8906 | 0.8500 | 0.8800 | 0.7857 | 0.8302 |

**Winner: Logistic Regression** — highest ROC-AUC (0.9531), simplest model, most interpretable.
For medical data, a well-tuned LR often beats tree models because the decision boundary is
roughly linear after feature scaling.

### MLflow Tracking

All runs logged to `mlruns/` with:
- Parameters: model type, hyperparameters
- Metrics: accuracy, precision, recall, F1, roc_auc
- Artifacts: confusion matrix PNG, ROC curve PNG, pipeline joblib file

To view the UI: `mlflow ui` → http://localhost:5000

**Total runs logged:** 9 (3 models × 3 run sessions during development)

### Plots Generated

| File | Contents |
|------|---------|
| `plots/cm_logistic_regression.png` | Confusion matrix — LR |
| `plots/cm_random_forest.png` | Confusion matrix — RF |
| `plots/cm_xgboost.png` | Confusion matrix — XGBoost |
| `plots/roc_logistic_regression.png` | ROC curve — LR (AUC=0.9531) |
| `plots/roc_random_forest.png` | ROC curve — RF (AUC=0.9342) |
| `plots/roc_xgboost.png` | ROC curve — XGBoost (AUC=0.8906) |

### Model Saved (`Task 4`)

- Format: `joblib` — faster than pickle for numpy arrays, standard sklearn ecosystem choice
- Path: `models/pipeline.pkl`
- Contains: full `sklearn.Pipeline` object (preprocessor + best classifier bundled)
- Reproducibility: `random_state=42` used everywhere, pinned versions in `requirements.txt`

---

## Issues Encountered & Fixed

- **MLflow `skops` trust error for XGBoost:** `mlflow.sklearn.log_model` uses `skops` under
  the hood and refuses to serialise XGBoost types without explicit trust. Fixed by saving the
  pipeline as a raw joblib artifact (`mlflow.log_artifact`) instead of using `log_model` — this
  works for all model types uniformly.

- **Unused import hint:** removed `mlflow.sklearn` and `mlflow.xgboost` imports after switching
  to joblib-based artifact logging.
