"""
Train Logistic Regression, Random Forest, and XGBoost on the Heart Disease dataset.
All runs tracked with MLflow. Best model saved to models/pipeline.pkl.

Usage:
    python src/train.py
    python src/train.py --quick-run   # fast smoke test for CI
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

warnings.filterwarnings("ignore")

# MLflow 3.x requires a database backend — file store is deprecated
MLFLOW_DB = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_confusion_matrix(y_true, y_pred, run_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix — {run_name}")
    path = PLOTS_DIR / f"cm_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_roc_curve(y_true, y_prob, run_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {run_name}")
    ax.legend()
    path = PLOTS_DIR / f"roc_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_precision_recall_curve(y_true, y_prob, run_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#e74c3c", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {run_name}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    path = PLOTS_DIR / f"pr_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_calibration_plot(y_true, y_prob, run_name):
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
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, "o-", color="#3498db", label="Training AUC")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15,
                    color="#3498db")
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", label="Validation AUC")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15,
                    color="#e74c3c")
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
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    # Get transformed feature names from the ColumnTransformer
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

    if hasattr(clf, "coef_"):
        importances = abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        return None

    # Take top 15 most important
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


def run_experiment(name, pipeline, params, X_train, y_train, X_test, y_test,
                   cv_score=None):
    mlflow.set_experiment("heart-disease-classification")
    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        if cv_score is not None:
            metrics["cv_roc_auc"] = cv_score

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        cm_path = save_confusion_matrix(y_test, y_pred, name)
        roc_path = save_roc_curve(y_test, y_prob, name)
        pr_path = save_precision_recall_curve(y_test, y_prob, name)
        cal_path = save_calibration_plot(y_test, y_prob, name)
        fi_path = save_feature_importance(pipeline, ALL_FEATURES, name)
        for path in [cm_path, roc_path, pr_path, cal_path]:
            mlflow.log_artifact(path)
        if fi_path:
            mlflow.log_artifact(fi_path)

        # Save pipeline as joblib artifact (works for all model types including XGBoost)
        tmp_path = MODELS_DIR / f"_tmp_{name}.pkl"
        joblib.dump(pipeline, tmp_path)
        mlflow.log_artifact(str(tmp_path), artifact_path="pipeline")
        tmp_path.unlink()

        cv_str = f"{cv_score:.4f}" if cv_score is not None else "N/A"
        print(f"[{name}] CV-AUC={cv_str}  Test-AUC={metrics['roc_auc']:.4f}"
              f"  Acc={metrics['accuracy']:.4f}")
        return metrics["roc_auc"], pipeline


def main(quick_run=False):
    df = load_data(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # --- Logistic Regression ---
    lr_pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    if quick_run:
        lr_best = lr_pipeline
        lr_params = {"model": "LogisticRegression", "C": 1.0, "cv_folds": 0}
        lr_cv_score = None
    else:
        gs = GridSearchCV(
            lr_pipeline,
            {"classifier__C": [0.01, 0.1, 1, 10, 100]},
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        gs.fit(X_train, y_train)
        lr_best = gs.best_estimator_
        lr_cv_score = gs.best_score_
        lr_params = {"model": "LogisticRegression", "cv_folds": 5, **gs.best_params_}

    auc, _ = run_experiment(
        "logistic_regression", lr_best, lr_params, X_train, y_train, X_test, y_test,
        cv_score=lr_cv_score
    )
    results.append(("LogisticRegression", auc, lr_best))

    # --- Random Forest ---
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=42))
    if quick_run:
        rf_best = rf_pipeline
        rf_params = {"model": "RandomForest", "n_estimators": 100, "cv_folds": 0}
        rf_cv_score = None
    else:
        rs = RandomizedSearchCV(
            rf_pipeline,
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10, 15],
                "classifier__min_samples_split": [2, 5, 10],
            },
            n_iter=10, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
        )
        rs.fit(X_train, y_train)
        rf_best = rs.best_estimator_
        rf_cv_score = rs.best_score_
        rf_params = {"model": "RandomForest", "cv_folds": 5, **rs.best_params_}

    auc, _ = run_experiment(
        "random_forest", rf_best, rf_params, X_train, y_train, X_test, y_test,
        cv_score=rf_cv_score
    )
    results.append(("RandomForest", auc, rf_best))

    # --- XGBoost ---
    xgb_pipeline = build_pipeline(XGBClassifier(eval_metric="logloss", random_state=42))
    if quick_run:
        xgb_best = xgb_pipeline
        xgb_params = {"model": "XGBoost", "n_estimators": 100, "cv_folds": 0}
        xgb_cv_score = None
    else:
        rs_xgb = RandomizedSearchCV(
            xgb_pipeline,
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
            },
            n_iter=8, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
        )
        rs_xgb.fit(X_train, y_train)
        xgb_best = rs_xgb.best_estimator_
        xgb_cv_score = rs_xgb.best_score_
        xgb_params = {"model": "XGBoost", "cv_folds": 5, **rs_xgb.best_params_}

    auc, _ = run_experiment(
        "xgboost", xgb_best, xgb_params, X_train, y_train, X_test, y_test,
        cv_score=xgb_cv_score
    )
    results.append(("XGBoost", auc, xgb_best))

    # --- Save best model ---
    best_name, best_auc, best_pipeline = max(results, key=lambda x: x[1])
    save_path = MODELS_DIR / "pipeline.pkl"
    joblib.dump(best_pipeline, save_path)
    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f}) saved to {save_path}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-run", action="store_true", help="Skip tuning (for CI)")
    args = parser.parse_args()
    main(quick_run=args.quick_run)
