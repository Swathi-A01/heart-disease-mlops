"""
Standalone inference script — run predictions from the command line
without needing the API running.

Usage examples:
    # Single patient from command line args
    python src/predict.py --age 67 --sex 1 --cp 4 --trestbps 160 --chol 286 \
        --fbs 0 --restecg 2 --thalach 108 --exang 1 --oldpeak 1.5 \
        --slope 2 --ca 3 --thal 7

    # Batch predictions from a CSV file
    python src/predict.py --input data/heart.csv --output predictions.csv

    # Use a specific model (default: models/pipeline.pkl)
    python src/predict.py --model models/pipeline.pkl --input data/heart.csv
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import engineer_features, CATEGORICAL_FEATURES

DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "pipeline.pkl"


def load_model(model_path: Path):
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("Run 'python src/train.py' first to train and save the model.")
        sys.exit(1)
    return joblib.load(model_path)


def predict_dataframe(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    df_feat = engineer_features(df)
    for col in CATEGORICAL_FEATURES:
        df_feat[col] = df_feat[col].astype(float)

    predictions = pipeline.predict(df_feat)
    probabilities = pipeline.predict_proba(df_feat)[:, 1]

    result = df.copy()
    result["prediction"] = predictions
    result["probability"] = probabilities.round(4)
    result["risk"] = ["high" if p == 1 else "low" for p in predictions]
    result["heart_rate_reserve"] = df_feat["heart_rate_reserve"].round(2)
    return result


def predict_single(pipeline, args) -> None:
    row = {
        "age": args.age, "sex": args.sex, "cp": args.cp,
        "trestbps": args.trestbps, "chol": args.chol, "fbs": args.fbs,
        "restecg": args.restecg, "thalach": args.thalach, "exang": args.exang,
        "oldpeak": args.oldpeak, "slope": args.slope, "ca": args.ca, "thal": args.thal,
    }
    df = pd.DataFrame([row])
    result = predict_dataframe(pipeline, df).iloc[0]

    risk_color = "\033[91m" if result["risk"] == "high" else "\033[92m"
    reset = "\033[0m"

    print("\n── Patient Risk Assessment ─────────────────────────────")
    print(f"  Prediction : {risk_color}{result['risk'].upper()} RISK{reset}")
    print(f"  Probability: {result['probability']:.1%} chance of heart disease")
    print(f"  Heart Rate Reserve: {result['heart_rate_reserve']:.1f} bpm")
    print("────────────────────────────────────────────────────────\n")


def predict_batch(pipeline, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df = df.dropna()  # drop rows with missing values before inference

    # Drop target if present (running inference on labelled data)
    if "target" in df.columns:
        ground_truth = df["target"].copy()
        df_input = df.drop(columns=["target"])
    else:
        ground_truth = None
        df_input = df

    result = predict_dataframe(pipeline, df_input)

    if ground_truth is not None:
        result["actual"] = ground_truth
        correct = (result["prediction"] == ground_truth).sum()
        accuracy = correct / len(result)
        print(f"Accuracy on provided labels: {accuracy:.1%} ({correct}/{len(result)})")

    high = (result["risk"] == "high").sum()
    low = (result["risk"] == "low").sum()

    result.to_csv(output_path, index=False)
    pct_high = high / len(result)
    pct_low = low / len(result)
    print("\nBatch prediction complete:")
    print(f"  Total patients : {len(result)}")
    print(f"  High risk      : {high} ({pct_high:.1%})")
    print(f"  Low risk       : {low} ({pct_low:.1%})")
    print(f"  Saved to       : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Heart Disease Risk Prediction — standalone inference"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to the saved pipeline.pkl")

    # Batch mode
    parser.add_argument("--input", type=Path,
                        help="CSV file for batch predictions")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"),
                        help="Output CSV path for batch predictions")

    # Single patient mode
    parser.add_argument("--age", type=float)
    parser.add_argument("--sex", type=int, choices=[0, 1])
    parser.add_argument("--cp", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--trestbps", type=float)
    parser.add_argument("--chol", type=float)
    parser.add_argument("--fbs", type=int, choices=[0, 1])
    parser.add_argument("--restecg", type=int, choices=[0, 1, 2])
    parser.add_argument("--thalach", type=float)
    parser.add_argument("--exang", type=int, choices=[0, 1])
    parser.add_argument("--oldpeak", type=float)
    parser.add_argument("--slope", type=int, choices=[1, 2, 3])
    parser.add_argument("--ca", type=float)
    parser.add_argument("--thal", type=float)

    args = parser.parse_args()
    pipeline = load_model(args.model)

    if args.input:
        predict_batch(pipeline, args.input, args.output)
    elif args.age is not None:
        predict_single(pipeline, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
