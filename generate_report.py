"""
Generate the MLOps Assignment 01 report as a professional Word document.
Run from the project root: python generate_report.py
"""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

PLOTS = Path("plots")
REPO_URL = "https://github.com/Swathi-A01/heart-disease-mlops"

# ── helpers ──────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14 if level == 1 else 8)
    h.paragraph_format.space_after = Pt(4)
    # Remove the ugly blue default heading colour — make it dark charcoal
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
    return h

def add_body(doc, text, space_after=6):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    for run in p.runs:
        run.font.size = Pt(11)
    return p

def add_image(doc, path, width=6.0, caption=None):
    if Path(path).exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width))
    if caption:
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(10)
        for run in cp.runs:
            run.font.size = Pt(9)
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1E, 0x1E, 0x2E)

def add_table(doc, headers, rows, header_color="1A1A2E", header_text_color="FFFFFF"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(
            int(header_text_color[0:2], 16),
            int(header_text_color[2:4], 16),
            int(header_text_color[4:6], 16),
        )
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_bg(cell, header_color)

    # Data rows
    for ri, row in enumerate(rows):
        tr = table.rows[ri + 1]
        bg = "F5F5F5" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(10)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_bg(cell, bg)

    doc.add_paragraph()
    return table


# ── document ─────────────────────────────────────────────────────────────────

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)

# ── COVER PAGE ───────────────────────────────────────────────────────────────

doc.add_paragraph()
doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Heart Disease Risk Prediction")
run.font.size = Pt(26)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("An End-to-End MLOps Pipeline")
run.font.size = Pt(16)
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()

meta_lines = [
    ("Course", "Machine Learning Operations (MLOps) — AIMLCZG523"),
    ("Assignment", "Assignment 01"),
    ("Student", "Swathi"),
    ("GitHub", REPO_URL),
    ("Dataset", "UCI Heart Disease (Cleveland)"),
]
for label, val in meta_lines:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    bold_run = p.add_run(f"{label}: ")
    bold_run.font.bold = True
    bold_run.font.size = Pt(11)
    val_run = p.add_run(val)
    val_run.font.size = Pt(11)

doc.add_page_break()

# ── 1. PROJECT OVERVIEW ───────────────────────────────────────────────────────

add_heading(doc, "1. Project Overview")

add_body(doc, (
    "This project delivers a production-grade machine learning system for predicting the risk "
    "of heart disease from clinical patient measurements. The work spans the complete MLOps lifecycle: "
    "raw data acquisition, exploratory analysis, feature engineering grounded in clinical knowledge, "
    "multi-model training with experiment tracking, a REST API, containerisation, Kubernetes "
    "deployment, and live monitoring. Every component is automated, tested, and reproducible from "
    "a single git clone."
))

add_body(doc, (
    "The dataset is the UCI Heart Disease (Cleveland) collection — 303 patients, 13 clinical "
    "features, and a binary target indicating presence or absence of heart disease. The problem "
    "is medically significant: early identification of at-risk patients allows timely intervention "
    "and is a well-studied classification benchmark."
))

add_heading(doc, "1.1 Technology Stack", level=2)

add_table(doc,
    ["Category", "Tool / Library"],
    [
        ["Language", "Python 3.11"],
        ["ML Framework", "scikit-learn 1.4, XGBoost 2.0"],
        ["Experiment Tracking", "MLflow 2.11 (SQLite backend)"],
        ["API Framework", "FastAPI 0.110 + Uvicorn"],
        ["Testing", "Pytest 8.1 — 24 tests"],
        ["Linting", "flake8 7.0"],
        ["CI/CD", "GitHub Actions"],
        ["Containerisation", "Docker 29.6 (python:3.11-slim)"],
        ["Orchestration", "Kubernetes via Docker Desktop"],
        ["Monitoring", "Prometheus + Grafana (docker-compose)"],
        ["Version Control", "Git + GitHub"],
    ]
)

# ── 2. SETUP & INSTALLATION ───────────────────────────────────────────────────

add_heading(doc, "2. Setup and Installation")

add_body(doc, "Prerequisites: Python 3.11, Git, Docker Desktop (with Kubernetes enabled).")

add_body(doc, "Clone and install:")
add_code(doc, "git clone https://github.com/Swathi-A01/heart-disease-mlops.git")
add_code(doc, "cd heart-disease-mlops")
add_code(doc, "python -m venv venv && source venv/bin/activate")
add_code(doc, "pip install -r requirements.txt")

add_body(doc, "Download the dataset:")
add_code(doc, "python data/download_data.py")

add_body(doc, "Train models and track experiments:")
add_code(doc, "python src/train.py")

add_body(doc, "Launch MLflow UI:")
add_code(doc, "mlflow ui --backend-store-uri sqlite:///mlflow.db")

add_body(doc, "Run the API locally:")
add_code(doc, "uvicorn api.main:app --reload   # http://localhost:8000/docs")

add_body(doc, "Run all 24 tests:")
add_code(doc, "pytest tests/ -v")

doc.add_page_break()

# ── 3. DATA ACQUISITION & EDA ─────────────────────────────────────────────────

add_heading(doc, "3. Data Acquisition and Exploratory Data Analysis")

add_body(doc, (
    "The dataset was downloaded programmatically from the UCI Machine Learning Repository via "
    "data/download_data.py. The raw Cleveland file contains no column headers and uses '?' to "
    "denote missing values. The download script adds the 14 column names, replaces '?' with NaN, "
    "and binarises the multi-class target (0–4 severity) into a binary label: 0 for no disease, "
    "1 for disease present. The cleaned file is saved to data/heart.csv."
))

add_heading(doc, "3.1 Missing Value Analysis", level=2)
add_body(doc, (
    "Six rows contained missing values: four in 'ca' (number of major vessels) and two in 'thal' "
    "(thalassemia type). At 2% of 303 rows, these were dropped rather than imputed — the dataset "
    "is small enough that imputation would introduce noise without meaningful benefit. The final "
    "working dataset contains 297 rows."
))
add_image(doc, PLOTS / "missing_values.png", width=5.5,
          caption="Figure 1. Missing values by feature. Red bars indicate affected columns.")

add_heading(doc, "3.2 Class Balance", level=2)
add_body(doc, (
    "The dataset is well-balanced: 160 patients without heart disease (53.9%) and 137 with it "
    "(46.1%). This near-even split means accuracy is a reliable metric and no resampling technique "
    "such as SMOTE is needed."
))
add_image(doc, PLOTS / "class_balance.png", width=5.5,
          caption="Figure 2. Class distribution showing near-equal split between classes.")

add_heading(doc, "3.3 Feature Distributions", level=2)
add_body(doc, (
    "Numeric features were examined as overlaid histograms split by target class. "
    "The most visually discriminating features are 'thalach' (maximum heart rate — "
    "disease patients achieve a lower maximum), 'oldpeak' (ST depression — elevated in disease), "
    "and 'ca' (vessel count — higher in disease). Age shows a moderate shift but is not a "
    "standalone strong predictor."
))
add_image(doc, PLOTS / "histograms_numeric.png", width=6.0,
          caption="Figure 3. Numeric feature distributions by class.")
add_image(doc, PLOTS / "histograms_categorical.png", width=6.0,
          caption="Figure 4. Categorical feature distributions by class.")

add_heading(doc, "3.4 Correlation Heatmap", level=2)
add_body(doc, (
    "Pearson correlations between all features and the target reveal the five strongest signals: "
    "thal (0.53), ca (0.46), thalach (−0.42), exang (0.42), and oldpeak (0.42). Notably, "
    "cholesterol shows almost no correlation with the target (0.08) in this dataset — a finding "
    "that contradicts the popular belief that high cholesterol alone predicts heart disease."
))
add_image(doc, PLOTS / "correlation_heatmap.png", width=6.0,
          caption="Figure 5. Feature correlation matrix (lower triangle).")

add_heading(doc, "3.5 Boxplots and Feature Relationships", level=2)
add_image(doc, PLOTS / "boxplots.png", width=6.0,
          caption="Figure 6. Numeric features vs target class (boxplots).")
add_image(doc, PLOTS / "oldpeak_cp_analysis.png", width=6.0,
          caption="Figure 7. ST depression (violin) and chest pain type disease rate.")
add_image(doc, PLOTS / "feature_relationships.png", width=6.0,
          caption="Figure 8. Age vs max heart rate scatter and cholesterol KDE by class.")

add_heading(doc, "3.6 Demographic Risk Analysis", level=2)
add_body(doc, (
    "Disease rate increases with age, peaking in the 60–70 bracket. The dataset is 68% male, "
    "which reflects the original clinical study demographics. Female patients in this dataset "
    "show a lower disease rate, though this partially reflects the smaller female sample size "
    "rather than a definitive biological finding."
))
add_image(doc, PLOTS / "demographic_risk.png", width=6.0,
          caption="Figure 9. Disease rate by age group and gender.")

add_heading(doc, "3.7 Parallel Coordinates", level=2)
add_body(doc, (
    "A parallel coordinates plot across the six most important numeric features shows a "
    "consistent visual pattern: disease patients (red) tend toward lower thalach, higher "
    "oldpeak and ca values. This multi-dimensional view confirms that no single feature is "
    "sufficient — the risk signal emerges from the combination."
))
add_image(doc, PLOTS / "parallel_coordinates.png", width=6.0,
          caption="Figure 10. Parallel coordinates — normalised patient profiles by class.")

doc.add_page_break()

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────

add_heading(doc, "4. Feature Engineering")

add_body(doc, (
    "Beyond the raw 13 features, five clinically motivated derived features were created "
    "in src/preprocess.py. Each is grounded in established medical guidelines."
))

add_table(doc,
    ["Feature", "Formula / Rule", "Clinical Basis"],
    [
        ["heart_rate_reserve", "(220 − age) − thalach",
         "Difference between predicted max HR and achieved max HR. Reduced reserve indicates impaired cardiac output."],
        ["age_thalach_ratio", "thalach / age",
         "Normalises achieved heart rate by age. Captures fitness relative to expected age-adjusted capacity."],
        ["st_slope_interaction", "oldpeak × slope",
         "Combines ST depression magnitude with the slope direction. Downsloping + high depression = high-risk signal."],
        ["bp_category", "JNC-8 stages 0–3",
         "Ordinal hypertension classification: 0=Normal (<120), 1=Elevated, 2=Stage1 HTN (130–139), 3=Stage2 HTN (≥140)."],
        ["chol_risk", "ATP III tiers 0–2",
         "Cholesterol risk tier: 0=Desirable (<200 mg/dl), 1=Borderline (200–239), 2=High (≥240)."],
    ]
)

add_image(doc, PLOTS / "engineered_features.png", width=6.0,
          caption="Figure 11. Engineered clinical features vs target class.")

add_heading(doc, "4.1 Preprocessing Pipeline", level=2)
add_body(doc, (
    "All feature transformations are encapsulated in a scikit-learn ColumnTransformer, "
    "which is then wrapped in a Pipeline together with the classifier. This is the correct "
    "MLOps pattern: the transformer is fitted once on training data only, and the identical "
    "fitted object is used at inference time, preventing data leakage."
))

add_table(doc,
    ["Feature Group", "Features", "Transformation"],
    [
        ["Numeric (9)", "age, trestbps, chol, thalach, oldpeak, ca,\nheart_rate_reserve, age_thalach_ratio, st_slope_interaction",
         "StandardScaler — zero mean, unit variance"],
        ["Categorical (6)", "cp, restecg, slope, thal, bp_category, chol_risk",
         "OneHotEncoder(drop='first') — removes dummy variable trap"],
        ["Binary (3)", "sex, fbs, exang",
         "Passthrough — already 0/1, no transformation needed"],
    ]
)

doc.add_page_break()

# ── 5. MODEL DEVELOPMENT ─────────────────────────────────────────────────────

add_heading(doc, "5. Model Development and Evaluation")

add_body(doc, (
    "Three classification algorithms were trained and compared: Logistic Regression, "
    "Random Forest, and XGBoost. Each model was hyperparameter-tuned using 5-fold "
    "StratifiedKFold cross-validation with ROC-AUC as the scoring metric. "
    "Stratified folds preserve the class ratio in each split, which is important for "
    "medical classification tasks."
))

add_heading(doc, "5.1 Hyperparameter Tuning", level=2)
add_table(doc,
    ["Model", "Tuning Method", "Search Space"],
    [
        ["Logistic Regression", "GridSearchCV",
         "C: [0.01, 0.1, 1, 10, 100]"],
        ["Random Forest", "RandomizedSearchCV (10 iter)",
         "n_estimators: [100,200,300], max_depth: [None,5,10,15], min_samples_split: [2,5,10]"],
        ["XGBoost", "RandomizedSearchCV (8 iter)",
         "n_estimators: [100,200], max_depth: [3,5,7], learning_rate: [0.01,0.1,0.2]"],
    ]
)

add_heading(doc, "5.2 Model Comparison", level=2)
add_table(doc,
    ["Model", "CV ROC-AUC", "Test ROC-AUC", "Accuracy", "Precision", "Recall", "F1"],
    [
        ["Logistic Regression", "0.8905", "0.9397", "0.850", "0.852", "0.821", "0.836"],
        ["Random Forest", "0.8866", "0.9464", "0.833", "0.875", "0.750", "0.808"],
        ["XGBoost", "0.8914", "0.9252", "0.850", "0.880", "0.786", "0.830"],
        ["Best → Random Forest", "—", "0.9464", "—", "—", "—", "—"],
    ]
)

add_body(doc, (
    "Random Forest achieved the highest test ROC-AUC of 0.9464 and was selected as the "
    "production model. The gap between CV AUC (~0.89) and test AUC (~0.94) is positive — "
    "the model generalises well and is not overfitting. Logistic Regression is a close "
    "second and remains competitive given its interpretability advantage."
))

add_heading(doc, "5.3 ROC Curves", level=2)
add_image(doc, PLOTS / "roc_random_forest.png", width=4.0,
          caption="Figure 12. ROC curve — Random Forest (AUC = 0.9464).")

add_heading(doc, "5.4 Confusion Matrices", level=2)
add_image(doc, PLOTS / "cm_random_forest.png", width=4.0,
          caption="Figure 13. Confusion matrix — Random Forest (test set).")

add_heading(doc, "5.5 Precision-Recall Curves", level=2)
add_body(doc, (
    "Precision-recall curves are particularly informative for clinical applications where "
    "false negatives (missed disease) carry a higher cost than false positives. The curves "
    "confirm that the Random Forest maintains high precision across a wide recall range."
))
add_image(doc, PLOTS / "pr_random_forest.png", width=4.0,
          caption="Figure 14. Precision-Recall curve — Random Forest.")

add_heading(doc, "5.6 Calibration and Feature Importance", level=2)
add_body(doc, (
    "Calibration plots assess whether predicted probabilities reflect true frequencies. "
    "A well-calibrated model is essential in medical settings — a prediction of 80% risk "
    "should correspond to 80% of those patients actually having disease."
))
add_image(doc, PLOTS / "calibration_random_forest.png", width=4.0,
          caption="Figure 15. Calibration plot — Random Forest.")
add_image(doc, PLOTS / "feature_importance_random_forest.png", width=5.5,
          caption="Figure 16. Feature importance — Random Forest (top 15 features).")

add_heading(doc, "5.7 Learning Curve", level=2)
add_body(doc, (
    "The learning curve confirms the model is not severely overfitting: training AUC converges "
    "toward validation AUC as data size increases. The validation score plateaus around 200 "
    "samples, suggesting the dataset is near its information capacity for this model class."
))
add_image(doc, PLOTS / "learning_curve_randomforest.png", width=5.5,
          caption="Figure 17. Learning curve — Random Forest.")

doc.add_page_break()

# ── 6. EXPERIMENT TRACKING ────────────────────────────────────────────────────

add_heading(doc, "6. Experiment Tracking with MLflow")

add_body(doc, (
    "MLflow 2.11 was used to track all training runs. Given that MLflow 3.x deprecated the "
    "file-based tracking store, a SQLite backend (mlflow.db) was configured via "
    "mlflow.set_tracking_uri(). This is the recommended approach for single-machine experiment "
    "tracking and avoids the alembic migration conflicts that affect the file store."
))

add_heading(doc, "6.1 What Was Logged Per Run", level=2)
add_table(doc,
    ["Item", "MLflow API", "Example Value"],
    [
        ["Model name", "log_param('model', ...)", "RandomForest"],
        ["Hyperparameters", "log_param(...)", "n_estimators=200, max_depth=10"],
        ["CV folds", "log_param('cv_folds', 5)", "5"],
        ["CV ROC-AUC", "log_metric('cv_roc_auc', ...)", "0.8866"],
        ["Test accuracy", "log_metric('accuracy', ...)", "0.833"],
        ["Test ROC-AUC", "log_metric('roc_auc', ...)", "0.9464"],
        ["Precision / Recall / F1", "log_metric(...)", "0.875 / 0.750 / 0.808"],
        ["Confusion matrix", "log_artifact(cm_path)", "plots/cm_random_forest.png"],
        ["ROC curve", "log_artifact(roc_path)", "plots/roc_random_forest.png"],
        ["PR curve", "log_artifact(pr_path)", "plots/pr_random_forest.png"],
        ["Calibration plot", "log_artifact(cal_path)", "plots/calibration_random_forest.png"],
        ["Feature importance", "log_artifact(fi_path)", "plots/feature_importance_random_forest.png"],
        ["Saved pipeline", "log_artifact(pkl_path)", "models/_tmp_random_forest.pkl"],
    ]
)

add_body(doc, "To launch the MLflow UI and inspect runs:")
add_code(doc, "mlflow ui --backend-store-uri sqlite:///mlflow.db")
add_body(doc, "Then navigate to http://localhost:5000 to view the heart-disease-classification experiment.")

doc.add_page_break()

# ── 7. MODEL PACKAGING ────────────────────────────────────────────────────────

add_heading(doc, "7. Model Packaging and Reproducibility")

add_body(doc, (
    "The production model is saved as a joblib-serialised sklearn Pipeline object at "
    "models/pipeline.pkl. Because the Pipeline bundles the ColumnTransformer preprocessor "
    "and the classifier together, a single joblib.load() call at inference time is sufficient "
    "to produce predictions — no separate preprocessing step or re-fitting is needed."
))

add_heading(doc, "7.1 Why joblib over pickle", level=2)
add_body(doc, (
    "joblib is the standard serialisation method in the scikit-learn ecosystem. It handles "
    "numpy arrays more efficiently than Python's built-in pickle by memory-mapping large arrays "
    "instead of copying them. For pipelines containing fitted transformers (which store "
    "numpy arrays internally), this results in faster load times and smaller file sizes."
))

add_heading(doc, "7.2 Reproducibility Guarantees", level=2)
add_table(doc,
    ["Mechanism", "How it Ensures Reproducibility"],
    [
        ["requirements.txt with pinned versions",
         "Exact dependency versions prevent library update from changing model behaviour"],
        ["random_state=42 everywhere",
         "train_test_split, all classifiers, and StratifiedKFold produce identical splits and models"],
        ["Preprocessor fitted on train set only",
         "No information from the test set leaks into scaling/encoding parameters"],
        ["engineer_features() in preprocess.py",
         "Derived features use the same deterministic formula at train time and inference time"],
        ["Docker image trains model at build time",
         "The saved .pkl and the sklearn version that loads it are guaranteed to match"],
    ]
)

doc.add_page_break()

# ── 8. CI/CD PIPELINE ─────────────────────────────────────────────────────────

add_heading(doc, "8. CI/CD Pipeline with GitHub Actions")

add_body(doc, (
    "A GitHub Actions workflow at .github/workflows/ci.yml runs automatically on every push "
    "to the main branch. The pipeline runs entirely on a fresh ubuntu-latest virtual machine, "
    "which proves that the code is reproducible in a clean environment with only the "
    "requirements.txt dependencies installed."
))

add_heading(doc, "8.1 Pipeline Steps", level=2)
add_table(doc,
    ["Step", "Tool", "Purpose"],
    [
        ["1. Checkout code", "actions/checkout@v4", "Fetch latest commit"],
        ["2. Set up Python 3.11", "actions/setup-python@v5", "Pin interpreter version"],
        ["3. Cache pip packages", "actions/cache@v4", "Speed up repeated runs"],
        ["4. Install dependencies", "pip install -r requirements.txt", "Reproduce environment"],
        ["5. Lint", "flake8 src/ api/ tests/ --max-line-length=100", "Catch style errors and unused imports"],
        ["6. Download dataset", "python data/download_data.py", "Fetch UCI data on clean VM"],
        ["7. Unit tests (preprocessing)", "pytest tests/test_preprocess.py --junitxml", "Verify data pipeline correctness"],
        ["8. Train model", "python src/train.py --quick-run", "Smoke test full training pipeline"],
        ["9. Unit tests (API)", "pytest tests/test_api.py --junitxml", "Verify API endpoints"],
        ["10. Upload test results", "actions/upload-artifact@v4", "JUnit XML saved per run"],
        ["11. Upload trained model", "actions/upload-artifact@v4", "pipeline.pkl saved per run"],
    ]
)

add_body(doc, (
    "The --quick-run flag passed to train.py skips hyperparameter tuning, reducing CI runtime "
    "to under 60 seconds while still exercising the complete data loading, preprocessing, "
    "training, MLflow logging, and model saving pathway."
))
add_body(doc, (
    "The pipeline is configured to fail hard on any error — a failed lint check, a single "
    "failing test, or a runtime error in train.py will stop the run immediately with a "
    "non-zero exit code and mark the commit as broken."
))

add_heading(doc, "8.2 Test Coverage", level=2)
add_table(doc,
    ["Test File", "Tests", "What Is Verified"],
    [
        ["test_preprocess.py", "10", "Data loading, no missing values, binary target, preprocessor shape,\npipeline fit/predict, engineered feature columns, HR reserve formula,\nBP/cholesterol category bounds, accuracy threshold"],
        ["test_api.py", "14", "Health endpoint, all response fields, binary output, probability range,\nrisk label consistency, 422 on invalid input, batch endpoint,\nbatch empty/overlimit validation, response time < 500ms"],
    ]
)

doc.add_page_break()

# ── 9. CONTAINERISATION ───────────────────────────────────────────────────────

add_heading(doc, "9. Model Containerisation with Docker")

add_body(doc, (
    "The application is packaged as a Docker image using python:3.11-slim as the base. "
    "The image is fully self-contained — it downloads the dataset and trains the model "
    "at build time, so the saved pipeline.pkl and the sklearn version that loads it are "
    "guaranteed to be identical, eliminating cross-version serialisation errors."
))

add_heading(doc, "9.1 Dockerfile Design", level=2)
add_code(doc, "FROM python:3.11-slim")
add_code(doc, "WORKDIR /app")
add_code(doc, "COPY requirements.txt .")
add_code(doc, "RUN pip install --no-cache-dir -r requirements.txt")
add_code(doc, "COPY src/ src/  &&  COPY data/ data/  &&  COPY api/ api/")
add_code(doc, "RUN python data/download_data.py")
add_code(doc, "RUN mkdir -p models plots && python src/train.py --quick-run")
add_code(doc, "EXPOSE 8000")
add_code(doc, "HEALTHCHECK CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\"")
add_code(doc, "CMD [\"uvicorn\", \"api.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]")

add_heading(doc, "9.2 API Endpoints", level=2)
add_table(doc,
    ["Endpoint", "Method", "Description"],
    [
        ["/health", "GET", "Liveness check — returns status and version"],
        ["/predict", "POST", "Single patient prediction — returns prediction, probability, risk, heart_rate_reserve, age_thalach_ratio"],
        ["/predict-batch", "POST", "Batch prediction (up to 100 patients) with aggregate summary"],
        ["/model-info", "GET", "Model type, feature list, dataset info"],
        ["/stats", "GET", "Live prediction counts and high-risk rate since startup"],
        ["/metrics", "GET", "Prometheus metrics — request count, latency histograms, custom counters"],
        ["/docs", "GET", "Auto-generated Swagger UI (FastAPI)"],
    ]
)

add_heading(doc, "9.3 Build and Run", level=2)
add_code(doc, "docker build -t heart-risk-api:latest .")
add_code(doc, "docker run -d -p 8000:8000 --name heart-risk heart-risk-api:latest")
add_code(doc, "curl http://localhost:8000/health")
add_code(doc, '# {"status":"ok","model":"heart-disease-classifier","version":"1.0.0"}')
add_code(doc, "curl -X POST http://localhost:8000/predict \\")
add_code(doc, '  -H "Content-Type: application/json" \\')
add_code(doc, '  -d \'{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,')
add_code(doc, '       "restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,')
add_code(doc, '       "slope":2,"ca":3,"thal":7}\'')
add_code(doc, '# {"prediction":1,"probability":0.9982,"risk":"high","heart_rate_reserve":45.0,"age_thalach_ratio":1.6119}')

doc.add_page_break()

# ── 10. KUBERNETES ────────────────────────────────────────────────────────────

add_heading(doc, "10. Production Deployment on Kubernetes")

add_body(doc, (
    "The containerised API was deployed to Kubernetes using Docker Desktop's built-in "
    "Kubernetes cluster. This simulates a production deployment without requiring a "
    "cloud account. The manifests at k8s/deployment.yaml and k8s/service.yaml define "
    "the desired state; Kubernetes continuously reconciles actual state to match."
))

add_heading(doc, "10.1 Deployment Configuration", level=2)
add_table(doc,
    ["Parameter", "Value", "Reason"],
    [
        ["Replicas", "2", "High availability — one pod crashing does not cause downtime"],
        ["Image pull policy", "IfNotPresent", "Use locally built image; no Docker Hub push required"],
        ["Memory request/limit", "256Mi / 512Mi", "Prevent OOM kills while allowing burst"],
        ["CPU request/limit", "250m / 500m", "Fair scheduling with headroom for inference spikes"],
        ["Readiness probe", "GET /health every 5s", "Pod only receives traffic when it is ready"],
        ["Liveness probe", "GET /health every 10s", "Kubernetes restarts unhealthy pods automatically"],
    ]
)

add_heading(doc, "10.2 Service Configuration", level=2)
add_body(doc, (
    "A LoadBalancer service exposes the deployment on port 80, routing to the container's "
    "port 8000. Docker Desktop maps LoadBalancer services to localhost automatically."
))

add_heading(doc, "10.3 Deploy Commands", level=2)
add_code(doc, "kubectl apply -f k8s/")
add_code(doc, "kubectl get pods")
add_code(doc, "# NAME                                     READY   STATUS    RESTARTS   AGE")
add_code(doc, "# heart-risk-deployment-7dc9f489b4-9vtl2   1/1     Running   0          2m")
add_code(doc, "# heart-risk-deployment-7dc9f489b4-dpnqv   1/1     Running   0          2m")
add_code(doc, "kubectl get services")
add_code(doc, "# NAME                 TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE")
add_code(doc, "# heart-risk-service   LoadBalancer   10.96.92.118   172.18.0.5    80:32373/TCP   8m")
add_code(doc, "curl http://localhost/predict -X POST -d '{...}'")
add_code(doc, '# {"prediction":1,"probability":0.9982,"risk":"high",...}')

doc.add_page_break()

# ── 11. MONITORING ────────────────────────────────────────────────────────────

add_heading(doc, "11. Monitoring and Logging")

add_body(doc, (
    "The API exposes a /metrics endpoint via prometheus-fastapi-instrumentator, which "
    "auto-instruments every HTTP endpoint with request count, latency histograms, and "
    "status code breakdown. Two custom Prometheus metrics were added for business-level "
    "monitoring specific to this application."
))

add_heading(doc, "11.1 Custom Prometheus Metrics", level=2)
add_table(doc,
    ["Metric", "Type", "Labels", "What It Measures"],
    [
        ["heart_risk_predictions_total", "Counter", "risk_level (high/low)",
         "Cumulative prediction count split by risk outcome"],
        ["heart_risk_prediction_duration_seconds", "Histogram", "—",
         "Per-prediction inference latency"],
        ["heart_risk_batch_size", "Histogram", "—",
         "Distribution of batch prediction sizes"],
        ["http_requests_total", "Counter (auto)", "handler, method, status",
         "Request count per endpoint and HTTP status"],
        ["http_request_duration_seconds", "Histogram (auto)", "handler",
         "End-to-end request latency per endpoint"],
    ]
)

add_heading(doc, "11.2 Structured Request Logging", level=2)
add_body(doc, "Every prediction request is logged in a structured format:")
add_code(doc, "2026-07-10 15:56:20 INFO PREDICT age=67 sex=1 cp=4 prob=0.9982 result=high hr_reserve=45.0")

add_heading(doc, "11.3 Monitoring Stack", level=2)
add_body(doc, "Prometheus and Grafana are started via docker-compose:")
add_code(doc, "cd monitoring && docker compose up -d")
add_body(doc, (
    "Prometheus scrapes /metrics every 15 seconds. Grafana connects to Prometheus as a "
    "data source and the provided grafana-dashboard.json imports a dashboard with four panels: "
    "prediction request rate, total predictions by risk level, latency p50/p95, and API success rate."
))

doc.add_page_break()

# ── 12. ARCHITECTURE ──────────────────────────────────────────────────────────

add_heading(doc, "12. System Architecture")

add_body(doc, (
    "The diagram below shows data flow across the full system. Each layer is independently "
    "deployable and testable. The preprocessing pipeline is the single point of feature "
    "transformation used by both the training path and the serving path."
))

# ASCII architecture diagram as code block
arch = """
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  UCI CSV ──► download_data.py ──► data/heart.csv                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
│  preprocess.py                                                   │
│  ColumnTransformer: StandardScaler + OneHotEncoder + passthrough │
│  engineer_features(): heart_rate_reserve, bp_category, etc.      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼─────────┐             ┌─────────▼─────────┐
│   TRAINING PATH   │             │   SERVING PATH    │
│  src/train.py     │             │  api/main.py      │
│  LR + RF + XGB    │             │  FastAPI          │
│  GridSearch / RS  │             │  /predict         │
│  MLflow logging   │             │  /predict-batch   │
│  pipeline.pkl ◄───┘             │  /model-info      │
└─────────┬─────────┘             │  /stats           │
          │ mlflow.db             └─────────┬─────────┘
┌─────────▼─────────┐                       │
│   MLFLOW UI       │             ┌─────────▼─────────┐
│  localhost:5000   │             │   DOCKER IMAGE    │
└───────────────────┘             │  python:3.11-slim │
                                  │  trains at build  │
                                  └─────────┬─────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │   KUBERNETES       │
                                  │  2 replicas        │
                                  │  LoadBalancer :80  │
                                  │  health probes     │
                                  └─────────┬─────────┘
                                            │
                        ┌───────────────────┴───────────────────┐
                        │                                       │
              ┌─────────▼─────────┐                 ┌──────────▼─────────┐
              │   PROMETHEUS      │                 │    GRAFANA         │
              │  scrapes /metrics │                 │  dashboard         │
              │  localhost:9090   │                 │  localhost:3000    │
              └───────────────────┘                 └────────────────────┘

              ┌──────────────────────────────────────────────────────────┐
              │                   CI/CD (GitHub Actions)                  │
              │  push ► lint ► download ► test ► train ► test API ► done │
              └──────────────────────────────────────────────────────────┘
"""
add_code(doc, arch)

doc.add_page_break()

# ── 13. REPOSITORY STRUCTURE ─────────────────────────────────────────────────

add_heading(doc, "13. Repository Structure")

add_code(doc, """heart-disease-mlops/
├── data/
│   ├── download_data.py        # fetches UCI dataset, adds headers, binarises target
│   └── heart.csv               # cleaned 297-row dataset
├── notebooks/
│   └── 01_eda.ipynb            # 12-section EDA with 17 visualisations
├── src/
│   ├── preprocess.py           # ColumnTransformer + engineer_features()
│   └── train.py                # 3 models, tuning, MLflow, 9+ plot types
├── api/
│   └── main.py                 # FastAPI: /predict /batch /stats /model-info /metrics
├── tests/
│   ├── test_preprocess.py      # 10 tests: data, features, formulas, accuracy threshold
│   └── test_api.py             # 14 tests: endpoints, batch, validation, response time
├── k8s/
│   ├── deployment.yaml         # 2 replicas, resource limits, health probes
│   └── service.yaml            # LoadBalancer port 80 → 8000
├── monitoring/
│   ├── prometheus.yml          # scrape config
│   ├── docker-compose.yml      # Prometheus + Grafana
│   └── grafana-dashboard.json  # 4-panel dashboard
├── plots/                      # 27 visualisations
├── models/                     # pipeline.pkl (gitignored, built in Docker)
├── process_tracking/           # step-by-step decision log (9 files)
├── .github/workflows/ci.yml    # CI: lint → test → train → test API → artifacts
├── Dockerfile                  # builds + trains model inside container
├── .dockerignore
├── pytest.ini
└── requirements.txt            # 16 pinned dependencies""")

doc.add_page_break()

# ── 14. REFERENCES ────────────────────────────────────────────────────────────

add_heading(doc, "14. References")

refs = [
    "Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304–310.",
    "UCI Machine Learning Repository — Heart Disease Dataset. https://archive.ics.uci.edu/ml/datasets/Heart+Disease",
    "Chobanian, A.V. et al. (2003). The Seventh Report of the Joint National Committee on Prevention, Detection, Evaluation, and Treatment of High Blood Pressure (JNC-7). JAMA, 289(19), 2560–2571.",
    "Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults (2001). Executive Summary of The Third Report of the National Cholesterol Education Program (NCEP) Expert Panel — ATP III. JAMA, 285(19), 2486–2497.",
    "Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems 28.",
    "MLflow Documentation. https://mlflow.org/docs/latest/index.html",
    "FastAPI Documentation. https://fastapi.tiangolo.com",
    "Kubernetes Documentation — Deployment and Service. https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
]

for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph(f"[{i}] {ref}")
    p.paragraph_format.space_after = Pt(4)
    for run in p.runs:
        run.font.size = Pt(10)

# ── SAVE ─────────────────────────────────────────────────────────────────────

out = Path("Heart_Disease_MLOps_Report.docx")
doc.save(out)
print(f"Report saved: {out}")
