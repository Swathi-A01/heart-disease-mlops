"""
data/download_data.py
─────────────────────
Downloads the Heart Disease UCI Dataset (Cleveland subset) and saves a
clean, analysis-ready CSV to data/heart.csv.

What this script does:
  1. Fetches the raw data file from the UCI ML Repository
  2. Attaches proper column headers (the raw file has none)
  3. Replaces '?' characters with NaN (UCI's encoding for missing values)
  4. Binarises the target column: original values are 0-4 (severity),
     we convert to 0 = no disease, 1 = disease present
  5. Saves the result and prints a summary
"""

import urllib.request
import pandas as pd
from pathlib import Path

# Source URL — Cleveland processed dataset from UCI ML Repository
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names in order — the raw file is headerless, so we define them manually
# based on the UCI dataset documentation
COLUMNS = [
    "age",      # age in years
    "sex",      # 1 = male, 0 = female
    "cp",       # chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic)
    "trestbps", # resting blood pressure in mmHg at hospital admission
    "chol",     # serum cholesterol in mg/dl
    "fbs",      # fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",  # resting ECG results (0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy)
    "thalach",  # maximum heart rate achieved during exercise
    "exang",    # exercise-induced angina (1=yes, 0=no)
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",    # slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)
    "ca",       # number of major vessels (0-3) coloured by fluoroscopy
    "thal",     # thalassemia: 3=normal, 6=fixed defect, 7=reversable defect
    "target",   # diagnosis: 0=no disease, 1-4=disease (we will binarise this)
]

# Output path — relative to this file's location so it works from any directory
OUTPUT = Path(__file__).parent / "heart.csv"


def download():
    """Download, clean, and save the dataset."""

    # Step 1: Download the raw file to a temporary location
    raw_path = Path(__file__).parent / "cleveland.data"
    print(f"Downloading from {URL}...")
    urllib.request.urlretrieve(URL, raw_path)

    # Step 2: Read into a DataFrame
    # - header=None because the file has no header row
    # - na_values="?" converts the UCI missing-value marker to proper NaN
    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")

    # Step 3: Binarise the target column
    # Original values are 0 (no disease) through 4 (severe disease).
    # For binary classification we only need: disease present (1) or absent (0).
    # Any value > 0 is mapped to 1.
    df["target"] = (df["target"] > 0).astype(int)

    # Step 4: Save cleaned CSV and delete the temporary raw file
    df.to_csv(OUTPUT, index=False)
    raw_path.unlink()  # clean up the raw download

    # Step 5: Print a summary so we can verify the download worked correctly
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(f"Missing values:\n{df.isnull().sum()}")  # expect ca=4, thal=2
    print(f"Class balance:\n{df['target'].value_counts()}")  # expect ~164 / 139


if __name__ == "__main__":
    download()
