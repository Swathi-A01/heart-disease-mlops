# Step 02 — Data Acquisition & EDA

**Date:** 2026-07-10
**Assignment Task:** Task 1 — Data Acquisition & EDA [5 marks]
**Status:** Complete

---

## What Was Done

### Dataset Download
- Script: `data/download_data.py`
- Source: UCI Machine Learning Repository (Cleveland subset)
- Raw file has no headers — added 14 column names manually in script
- Raw file uses `?` for missing values — replaced with NaN on load
- Target column is 0–4 (severity) — binarized to 0/1 (no disease / disease present)
- Output: `data/heart.csv` — 303 rows, 14 columns

**Download result:**
```
303 rows saved
Missing values: ca=4, thal=2  (6 total)
Class balance: 0→164, 1→139
```

### EDA Notebook: `notebooks/01_eda.ipynb`

Executed with `jupyter nbconvert --execute` — all outputs embedded in the notebook file.

#### Sections built:

| Section | Plot saved |
|---------|-----------|
| Missing value bar chart | `plots/missing_values.png` |
| Class balance (bar + pie) | `plots/class_balance.png` |
| Numeric histograms overlaid by target | `plots/histograms_numeric.png` |
| Categorical bar charts by target | `plots/histograms_categorical.png` |
| Correlation heatmap (lower triangle) | `plots/correlation_heatmap.png` |
| Boxplots — numeric features vs target | `plots/boxplots.png` |
| Scatter (age vs thalach) + KDE (chol) | `plots/feature_relationships.png` |
| Violin (oldpeak) + CP heatmap | `plots/oldpeak_cp_analysis.png` |

---

## Key EDA Findings (for the report)

1. **Class balance:** 54.3% No Disease / 45.7% Disease — near-perfect. No SMOTE needed.

2. **Missing values:** Only 6 rows (2%) had NaN in `ca` and `thal`. Dropped safely — dataset remains 297 rows.

3. **Strongest features correlated with target:**
   - `thalach` (max heart rate): negative — disease patients have lower max HR
   - `cp` (chest pain type): positive — asymptomatic type (3) has highest disease rate
   - `oldpeak` (ST depression): positive — clearly elevated in disease cases
   - `exang` (exercise angina): positive — expected clinical predictor
   - `ca` (vessel count): positive — more blocked vessels = higher disease probability

4. **Surprising finding:** `cp=3` (asymptomatic) has the *highest* disease rate. Medically known as silent heart disease — patients feel no pain but are at high risk.

5. **Age:** slightly higher in disease patients on average but not a strong standalone predictor.

6. **Cholesterol:** surprisingly similar distribution between both classes — not a strong signal here alone.

---

## Issues Encountered & Fixed

- **seaborn palette dict error:** Newer seaborn (0.13+) requires string keys in palette dicts
  when the x-axis column is numeric. Fixed by creating `df_str` copy with target cast to string,
  and using `{'0': color, '1': color}` keys for boxplot and violin plot cells.
