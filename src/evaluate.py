"""
Model evaluation script — generates a full comparison report across all
trained models logged in MLflow, saves results to a CSV and prints a summary.

Usage:
    python src/evaluate.py
    python src/evaluate.py --output reports/model_comparison.csv
    python src/evaluate.py --experiment heart-disease-classification
"""
import argparse
import sys
from pathlib import Path

import joblib
import mlflow

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    classification_report,
)
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import ALL_FEATURES, TARGET, load_data

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "pipeline.pkl"
MLFLOW_DB = Path(__file__).parent.parent / "mlflow.db"


def fetch_mlflow_runs(experiment_name: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    try:
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        if runs.empty:
            return pd.DataFrame()
        cols = [
            "tags.mlflow.runName",
            "metrics.roc_auc", "metrics.cv_roc_auc",
            "metrics.accuracy", "metrics.precision",
            "metrics.recall", "metrics.f1",
            "params.model", "params.cv_folds",
        ]
        available = [c for c in cols if c in runs.columns]
        return runs[available].rename(columns={
            "tags.mlflow.runName": "run_name",
            "metrics.roc_auc": "test_roc_auc",
            "metrics.cv_roc_auc": "cv_roc_auc",
            "metrics.accuracy": "accuracy",
            "metrics.precision": "precision",
            "metrics.recall": "recall",
            "metrics.f1": "f1",
            "params.model": "model_type",
            "params.cv_folds": "cv_folds",
        })
    except Exception as e:
        print(f"Could not fetch MLflow runs: {e}")
        return pd.DataFrame()


def evaluate_saved_model(model_path: Path, data_path: Path) -> dict:
    """Run fresh cross-validation on the saved best model."""
    df = load_data(data_path)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    pipeline = joblib.load(model_path)
    clf_name = type(pipeline.named_steps["classifier"]).__name__

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=True,
    )

    # also get test set metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": clf_name,
        "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
        "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "cv_f1_mean": cv_results["test_f1"].mean(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Disease", "Disease"]),
    }


def print_summary(mlflow_runs: pd.DataFrame, fresh_eval: dict) -> None:
    print("\n" + "=" * 65)
    print(" MODEL EVALUATION REPORT — Heart Disease Risk Prediction")
    print("=" * 65)

    if not mlflow_runs.empty:
        # Get best run per model type (by test_roc_auc)
        tuned = mlflow_runs[mlflow_runs["cv_folds"].fillna("0") != "0"]
        if tuned.empty:
            tuned = mlflow_runs

        print("\nMLflow Tracked Runs (tuned, sorted by test ROC-AUC):")
        print(f"  {'Model':<22} {'CV AUC':>8} {'Test AUC':>9} {'Accuracy':>9} {'F1':>8}")
        print("  " + "-" * 60)

        # deduplicate by model_type, keep best test_roc_auc
        best = tuned.sort_values("test_roc_auc", ascending=False)
        seen = set()
        for _, row in best.iterrows():
            model = row.get("model_type", row.get("run_name", "?"))
            if model in seen:
                continue
            seen.add(model)
            cv_val = row.get("cv_roc_auc", float("nan"))
            cv = f"{cv_val:.4f}" if pd.notna(cv_val) else "  N/A "
            print(f"  {str(model):<22} {cv:>8} "
                  f"{row.get('test_roc_auc', 0):.4f}    "
                  f"{row.get('accuracy', 0):.4f}   "
                  f"{row.get('f1', 0):.4f}")

    print(f"\nFresh Cross-Validation on Saved Model ({fresh_eval['model']}):")
    print(f"  CV Accuracy    : {fresh_eval['cv_accuracy_mean']:.4f}")
    print(f"  CV ROC-AUC     : {fresh_eval['cv_roc_auc_mean']:.4f}")
    print(f"  CV F1          : {fresh_eval['cv_f1_mean']:.4f}")
    print("\nHeld-out Test Set:")
    print("  Test Accuracy  : {:.4f}".format(fresh_eval['test_accuracy']))
    print("  Test ROC-AUC   : {:.4f}".format(fresh_eval['test_roc_auc']))
    print("  Test Precision : {:.4f}".format(fresh_eval['test_precision']))
    print("  Test Recall    : {:.4f}".format(fresh_eval['test_recall']))
    print("  Test F1        : {:.4f}".format(fresh_eval['test_f1']))
    print("\nClassification Report (test set):")
    for line in fresh_eval["classification_report"].split("\n"):
        print("  " + line)
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models and generate comparison report"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison.csv"),
                        help="Path to save the CSV comparison")
    parser.add_argument("--experiment", default="heart-disease-classification",
                        help="MLflow experiment name")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print("Dataset not found. Run: python data/download_data.py")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train.py")
        sys.exit(1)

    print("Fetching MLflow runs...")
    mlflow_runs = fetch_mlflow_runs(args.experiment)

    print("Running fresh cross-validation on saved model...")
    fresh_eval = evaluate_saved_model(MODEL_PATH, DATA_PATH)

    print_summary(mlflow_runs, fresh_eval)

    # Save CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": fresh_eval["model"],
        "cv_accuracy": round(fresh_eval["cv_accuracy_mean"], 4),
        "cv_roc_auc": round(fresh_eval["cv_roc_auc_mean"], 4),
        "test_accuracy": round(fresh_eval["test_accuracy"], 4),
        "test_roc_auc": round(fresh_eval["test_roc_auc"], 4),
        "test_precision": round(fresh_eval["test_precision"], 4),
        "test_recall": round(fresh_eval["test_recall"], 4),
        "test_f1": round(fresh_eval["test_f1"], 4),
    }

    if not mlflow_runs.empty:
        mlflow_runs.to_csv(args.output, index=False)
        print(f"Full MLflow run comparison saved to: {args.output}")
    else:
        pd.DataFrame([row]).to_csv(args.output, index=False)
        print(f"Evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
