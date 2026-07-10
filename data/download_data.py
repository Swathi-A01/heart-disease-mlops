"""
Downloads the Heart Disease UCI Dataset (Cleveland) and saves it with proper headers.
Source: UCI Machine Learning Repository
"""
import urllib.request
import pandas as pd
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

OUTPUT = Path(__file__).parent / "heart.csv"


def download():
    raw_path = Path(__file__).parent / "cleveland.data"
    print(f"Downloading from {URL}...")
    urllib.request.urlretrieve(URL, raw_path)

    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(OUTPUT, index=False)
    raw_path.unlink()

    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Class balance:\n{df['target'].value_counts()}")


if __name__ == "__main__":
    download()
