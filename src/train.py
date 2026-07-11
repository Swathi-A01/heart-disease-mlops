"""
src/train.py
────────────
Trains three classification models on the Heart Disease UCI dataset,
tracks every experiment in MLflow, and saves the best model to disk.

Models trained:
  1. Logistic Regression  — tuned with GridSearchCV over C values
  2. Random Forest        — tuned with RandomizedSearchCV
  3. XGBoost              — tuned with RandomizedSearchCV

All runs log: parameters, metrics (accuracy, precision, recall, F1, ROC-AUC,
CV-AUC), and artifacts (confusion matrix, ROC curve, PR curve, calibration
plot, feature importance, pipeline .pkl).

Usage:
    python src/train.py               # full run with hyperparameter tuning
    python src/train.py --quick-run   # no tuning — used by CI pipeline
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from xgboost import XGBClassifier

from preprocess import ALL_FEATURES, TARGET, build_pipeline, load_data

# Suppress non-critical sklearn and XGBoost warnings during training
warnings.filterwarnings("ignore")

# MLflow 3.x dropped the file-based tracking store.
# We point to a SQLite database so all experiment data persists between runs.
# View with: mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_DB = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

# Resolve paths relative to this file so the script works from any directory
DATA_PATH  = Path(__file__).parent.parent / "data" / "heart.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR  = Path(__file__).parent.parent / "plots"

# Create output directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 5 evaluation metrics in one call."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, run_name):
    """
    Plot and save the confusion matrix for a model run.
    Shows TP, TN, FP, FN counts — useful for understanding where the model fails.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix — {run_name}")
    path = PLOTS_DIR / f"cm_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_roc_curve(y_true, y_prob, run_name):
    """
    Plot and save the ROC curve (True Positive Rate vs False Positive Rate).
    AUC closer to 1.0 = better discrimination between classes.
    The diagonal dashed line = random classifier (AUC = 0.5).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")  # diagonal = random baseline
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {run_name}")
    ax.legend()
    path = PLOTS_DIR / f"roc_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_precision_recall_curve(y_true, y_prob, run_name):
    """
    Plot and save the Precision-Recall curve.
    More informative than ROC for imbalanced datasets.
    For medical screening, recall (catching all disease cases) is prioritised.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#e74c3c", lw=2)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title(f"Precision-Recall Curve — {run_name}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    path = PLOTS_DIR / f"pr_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_calibration_plot(y_true, y_prob, run_name):
    """
    Plot and save the calibration curve.
    A perfectly calibrated model lies on the diagonal — predicted probability of 0.7
    should mean 70% of those patients actually have disease.
    Important in clinical settings where probability scores drive decisions.
    """
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_predicted, fraction_of_positives, "s-", label=run_name, color="#2ecc71")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Plot — {run_name}")
    ax.legend(fontsize=9)
    path = PLOTS_DIR / f"calibration_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_learning_curve(pipeline, X, y, run_name):
    """
    Plot and save the learning curve (performance vs training set size).
    Shows whether the model would benefit from more data.
    A small gap between train/val AUC = model is well-generalised.
    A large gap = overfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=5,                               # 5-fold cross-validation for each point
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8),  # 8 evenly spaced training set sizes
        n_jobs=-1                           # use all CPU cores
    )
    # Average across the 5 folds for each training size
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    # Shaded regions show ±1 std dev across folds
    ax.plot(train_sizes, train_mean, "o-", color="#3498db", label="Training AUC")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#3498db")
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", label="Validation AUC")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning Curve — {run_name}")
    ax.legend()
    ax.set_ylim([0.5, 1.05])
    path = PLOTS_DIR / f"learning_curve_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_feature_importance(pipeline, feature_names, run_name):
    """
    Plot and save feature importance for the given pipeline.
    Works with both tree-based models (feature_importances_) and
    linear models (coef_). Returns None if neither is available.
    """
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    # Reconstruct feature names after OneHotEncoding expands categorical columns
    try:
        ohe_features = pre.named_transformers_["cat"].get_feature_names_out(
            ["cp", "restecg", "slope", "thal", "bp_category", "chol_risk"]
        ).tolist()
    except Exception:
        ohe_features = []

    num_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca",
                    "heart_rate_reserve", "age_thalach_ratio", "st_slope_interaction"]
    bin_features = ["sex", "fbs", "exang"]
    transformed_names = num_features + ohe_features + bin_features

    # Extract importances — different attribute for linear vs tree models
    if hasattr(clf, "coef_"):
        # Logistic Regression: use absolute value of coefficients
        importances = abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        # Random Forest / XGBoost: use built-in importance scores
        importances = clf.feature_importances_
    else:
        return None  # model type not supported

    # Show only the top 15 features to keep the plot readable
    n = min(15, len(importances))
    idx = importances.argsort()[-n:][::-1]
    top_names = [transformed_names[i] if i < len(transformed_names) else f"f{i}"
                 for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(n), top_vals[::-1], color="#9b59b6", edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (Top {n}) — {run_name}")
    plt.tight_layout()
    path = PLOTS_DIR / f"feature_importance_{run_name}.png"
    plt.savefig(path)
    plt.close()
    return str(path)


# ── MLflow experiment runner ──────────────────────────────────────────────────

def run_experiment(name, pipeline, params, X_train, y_train, X_test, y_test,
                   cv_score=None):
    """
    Train a pipeline, evaluate it, log everything to MLflow, and return test ROC-AUC.

    What gets logged per run:
      - Params : model name, hyperparameters, cv_folds
      - Metrics: accuracy, precision, recall, f1, roc_auc, cv_roc_auc (if available)
      - Artifacts: confusion matrix, ROC curve, PR curve, calibration plot,
                   feature importance, fitted pipeline (.pkl)
    """
    mlflow.set_experiment("heart-disease-classification")
    with mlflow.start_run(run_name=name):

        # Train the full pipeline (preprocessor + classifier) on training data
        pipeline.fit(X_train, y_train)

        # Generate predictions on the held-out test set
        y_pred = pipeline.predict(X_test)           # hard labels (0 or 1)
        y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of disease

        # Compute all evaluation metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Optionally add the cross-validation score from hyperparameter tuning
        if cv_score is not None:
            metrics["cv_roc_auc"] = cv_score

        # Log everything to MLflow
        mlflow.log_params(params)   # hyperparameters
        mlflow.log_metrics(metrics) # evaluation scores

        # Save and log all visualisation plots as artifacts
        cm_path  = save_confusion_matrix(y_test, y_pred, name)
        roc_path = save_roc_curve(y_test, y_prob, name)
        pr_path  = save_precision_recall_curve(y_test, y_prob, name)
        cal_path = save_calibration_plot(y_test, y_prob, name)
        fi_path  = save_feature_importance(pipeline, ALL_FEATURES, name)

        for path in [cm_path, roc_path, pr_path, cal_path]:
            mlflow.log_artifact(path)
        if fi_path:
            mlflow.log_artifact(fi_path)

        # Save the fitted pipeline as a joblib artifact.
        # Using joblib instead of mlflow.sklearn.log_model because the latter
        # has a known issue with XGBoost types in newer versions (skops trust error).
        tmp_path = MODELS_DIR / f"_tmp_{name}.pkl"
        joblib.dump(pipeline, tmp_path)
        mlflow.log_artifact(str(tmp_path), artifact_path="pipeline")
        tmp_path.unlink()  # remove the temporary file

        cv_str = f"{cv_score:.4f}" if cv_score is not None else "N/A"
        print(f"[{name}] CV-AUC={cv_str}  Test-AUC={metrics['roc_auc']:.4f}"
              f"  Acc={metrics['accuracy']:.4f}")
        return metrics["roc_auc"], pipeline


# ── Main training pipeline ────────────────────────────────────────────────────

def main(quick_run=False):
    """
    Main training function.

    quick_run=True (--quick-run flag): skips hyperparameter tuning, uses defaults.
    Used by the CI pipeline to verify the code runs end-to-end without errors.

    quick_run=False (default): runs full GridSearch/RandomizedSearch tuning.
    Takes ~5-10 minutes but finds the best hyperparameters.
    """

    # Load data with feature engineering applied
    df = load_data(DATA_PATH)
    X = df[ALL_FEATURES]  # all 18 features (13 raw + 5 derived)
    y = df[TARGET]        # binary target: 0 or 1

    # Split: 80% train, 20% test
    # stratify=y ensures both splits have the same class ratio (~54%/46%)
    # random_state=42 makes the split reproducible across runs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StratifiedKFold for cross-validation during hyperparameter search
    # Stratified = each fold preserves the class ratio
    # 5 folds = 5 train/validation splits, scores averaged for stability
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []  # will store (model_name, test_roc_auc, fitted_pipeline)

    # ── Model 1: Logistic Regression ─────────────────────────────────────────
    # Linear model — works well when decision boundary is approximately linear.
    # C is the inverse regularisation strength: small C = more regularisation.
    lr_pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))

    if quick_run:
        # CI mode: use default C=1.0, skip grid search
        lr_best     = lr_pipeline
        lr_params   = {"model": "LogisticRegression", "C": 1.0, "cv_folds": 0}
        lr_cv_score = None
    else:
        # Full mode: try 5 different C values with 5-fold CV, pick the best
        gs = GridSearchCV(
            lr_pipeline,
            {"classifier__C": [0.01, 0.1, 1, 10, 100]},
            cv=cv, scoring="roc_auc", n_jobs=-1  # n_jobs=-1 = use all CPU cores
        )
        gs.fit(X_train, y_train)
        lr_best     = gs.best_estimator_  # the pipeline with best C value
        lr_cv_score = gs.best_score_      # best mean CV ROC-AUC across 5 folds
        lr_params   = {"model": "LogisticRegression", "cv_folds": 5, **gs.best_params_}

    auc, _ = run_experiment(
        "logistic_regression", lr_best, lr_params, X_train, y_train, X_test, y_test,
        cv_score=lr_cv_score
    )
    results.append(("LogisticRegression", auc, lr_best))

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    # Ensemble of decision trees — can capture non-linear feature interactions.
    # RandomizedSearchCV samples n_iter random combinations from the search space,
    # faster than trying every combination (GridSearchCV).
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=42))

    if quick_run:
        rf_best     = rf_pipeline
        rf_params   = {"model": "RandomForest", "n_estimators": 100, "cv_folds": 0}
        rf_cv_score = None
    else:
        rs = RandomizedSearchCV(
            rf_pipeline,
            {
                "classifier__n_estimators":    [100, 200, 300],  # number of trees
                "classifier__max_depth":       [None, 5, 10, 15], # max tree depth (None=unlimited)
                "classifier__min_samples_split": [2, 5, 10],      # min samples to split a node
            },
            n_iter=10, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
        )
        rs.fit(X_train, y_train)
        rf_best     = rs.best_estimator_
        rf_cv_score = rs.best_score_
        rf_params   = {"model": "RandomForest", "cv_folds": 5, **rs.best_params_}

    auc, _ = run_experiment(
        "random_forest", rf_best, rf_params, X_train, y_train, X_test, y_test,
        cv_score=rf_cv_score
    )
    results.append(("RandomForest", auc, rf_best))

    # ── Model 3: XGBoost ──────────────────────────────────────────────────────
    # Gradient boosting — builds trees sequentially, each correcting the previous.
    # eval_metric="logloss" suppresses a deprecation warning.
    xgb_pipeline = build_pipeline(XGBClassifier(eval_metric="logloss", random_state=42))

    if quick_run:
        xgb_best     = xgb_pipeline
        xgb_params   = {"model": "XGBoost", "n_estimators": 100, "cv_folds": 0}
        xgb_cv_score = None
    else:
        rs_xgb = RandomizedSearchCV(
            xgb_pipeline,
            {
                "classifier__n_estimators":  [100, 200],       # boosting rounds
                "classifier__max_depth":     [3, 5, 7],        # tree depth per round
                "classifier__learning_rate": [0.01, 0.1, 0.2], # step size (shrinkage)
            },
            n_iter=8, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
        )
        rs_xgb.fit(X_train, y_train)
        xgb_best     = rs_xgb.best_estimator_
        xgb_cv_score = rs_xgb.best_score_
        xgb_params   = {"model": "XGBoost", "cv_folds": 5, **rs_xgb.best_params_}

    auc, _ = run_experiment(
        "xgboost", xgb_best, xgb_params, X_train, y_train, X_test, y_test,
        cv_score=xgb_cv_score
    )
    results.append(("XGBoost", auc, xgb_best))

    # ── Select and save the best model ────────────────────────────────────────
    # Compare all three by test ROC-AUC — highest wins
    best_name, best_auc, best_pipeline = max(results, key=lambda x: x[1])
    save_path = MODELS_DIR / "pipeline.pkl"

    # joblib.dump saves the entire pipeline: fitted scaler + encoder + classifier
    # This single file is all that's needed for inference
    joblib.dump(best_pipeline, save_path)
    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f}) saved to {save_path}")

    # Generate and log the learning curve for the winning model (full run only)
    if not quick_run:
        print("Generating learning curve for best model...")
        X_all = df[ALL_FEATURES]
        y_all = df[TARGET]
        lc_path = save_learning_curve(best_pipeline, X_all, y_all, best_name.lower())
        mlflow.set_experiment("heart-disease-classification")
        with mlflow.start_run(run_name=f"{best_name.lower()}_learning_curve"):
            mlflow.log_artifact(lc_path)
        print(f"Learning curve logged to MLflow: {lc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train heart disease classifiers with MLflow tracking"
    )
    parser.add_argument(
        "--quick-run", action="store_true",
        help="Skip hyperparameter tuning — used by CI pipeline for speed"
    )
    args = parser.parse_args()
    main(quick_run=args.quick_run)
