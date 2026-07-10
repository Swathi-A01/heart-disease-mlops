# Step 01 — Repository Setup & Project Scaffold

**Date:** 2026-07-10  
**Assignment Task:** Foundation (all tasks depend on this)  
**Status:** Complete

---

## What Was Done

### 1. GitHub Repository Created
- **Repo name:** `heart-disease-mlops`
- **URL:** https://github.com/Swathi-A01/heart-disease-mlops
- **Visibility:** Public (required for assignment submission)
- **Tool used:** `gh repo create` via GitHub CLI (already authenticated as Swathi-A01)

### 2. Folder Structure Scaffolded

```
heart-disease-mlops/
├── data/                    # Raw dataset + download script
├── notebooks/               # Jupyter EDA notebook
├── src/                     # Core Python source code
│   ├── preprocess.py        # Reusable sklearn pipeline
│   └── train.py             # Training + MLflow tracking
├── api/                     # FastAPI serving app
│   └── main.py
├── tests/                   # Pytest unit tests
│   ├── test_preprocess.py
│   └── test_api.py
├── k8s/                     # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/              # Prometheus + Grafana config
│   ├── prometheus.yml
│   └── docker-compose.yml
├── .github/workflows/       # GitHub Actions CI/CD
│   └── ci.yml
├── plots/                   # Auto-generated training plots (gitignored)
├── models/                  # Saved model pipeline (gitignored)
├── screenshots/             # Report screenshots
├── process_tracking/        # This folder — step-by-step logs
├── Dockerfile
├── requirements.txt
└── README.md
```

### 3. Core Files Created

#### `requirements.txt`
Pinned exact versions for full reproducibility:
- `pandas==2.2.0`, `numpy==1.26.4` — data handling
- `scikit-learn==1.4.0`, `xgboost==2.0.3` — ML models
- `mlflow==2.11.0` — experiment tracking
- `fastapi==0.110.0`, `uvicorn==0.29.0` — API serving
- `prometheus-fastapi-instrumentator==6.1.0` — monitoring
- `pytest==8.1.1`, `httpx==0.27.0` — testing

#### `.gitignore`
Excludes: `mlruns/`, `models/*.pkl`, `data/*.csv`, `.env`, `__pycache__/`  
Reason: model binaries and datasets don't belong in git (too large, versioned separately via MLflow)

#### `data/download_data.py`
- Downloads UCI Cleveland Heart Disease dataset directly from UCI repo
- Adds proper column headers (raw file has none)
- Replaces `?` missing values with NaN and drops affected rows (only 6 rows — safe to drop)
- Binarizes target: original is 0–4 (severity), we convert to 0/1 (no disease / disease)
- Saves clean CSV to `data/heart.csv`

#### `src/preprocess.py`
Defines reusable sklearn `ColumnTransformer`:
- **Numeric features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) → `StandardScaler`
- **Categorical features** (`cp`, `restecg`, `slope`, `thal`) → `OneHotEncoder(drop='first')`
- **Binary features** (`sex`, `fbs`, `exang`) → passthrough (already 0/1, no transformation needed)

Key design decision: wrapping this in `build_pipeline(classifier)` means the exact same
preprocessing is used at training time AND inference time — no train/test leakage possible.

#### `src/train.py`
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Uses GridSearchCV (LR) and RandomizedSearchCV (RF, XGBoost) for tuning
- All runs logged to MLflow with params + metrics + confusion matrix + ROC curve artifacts
- Saves best model (highest ROC-AUC) to `models/pipeline.pkl` using joblib
- `--quick-run` flag disables tuning for fast CI smoke test

#### `api/main.py`
FastAPI app with:
- `POST /predict` — accepts 13 patient features, returns prediction + probability + risk label
- `GET /health` — liveness check
- `GET /metrics` — Prometheus metrics (auto-instrumented)
- Structured logging of every prediction request

#### `tests/`
- `test_preprocess.py` — tests data loading, missing values, target binarization, pipeline shape
- `test_api.py` — tests /health, /predict valid, /predict invalid input (422 validation)

#### `.github/workflows/ci.yml`
Pipeline steps:
1. Install dependencies
2. `flake8` lint
3. Download dataset
4. Run `test_preprocess.py`
5. Train model (quick run)
6. Run `test_api.py`
7. Upload trained model as workflow artifact

#### `Dockerfile`
- Base: `python:3.11-slim` (minimal image, ~50% smaller than full python)
- Copies: `models/`, `api/`, `src/`, `requirements.txt`
- Exposes port 8000
- CMD: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

#### `k8s/deployment.yaml` + `k8s/service.yaml`
- Deployment: 2 replicas, resource limits set (256Mi/500m)
- Service: LoadBalancer type, port 80 → container 8000
- `imagePullPolicy: Never` — tells Kubernetes to use local Docker image (required for Minikube)

#### `monitoring/`
- `prometheus.yml` — scrapes API at `host.docker.internal:8000/metrics` every 15s
- `docker-compose.yml` — runs Prometheus (9090) + Grafana (3000) as containers

---

## Environment Check (done before scaffold)

| Tool | Status |
|------|--------|
| Docker Desktop | Installed + running (v29.6.1) |
| Docker Kubernetes | Not enabled yet — needs to be turned on in Docker Desktop settings |
| Minikube | Not installed |
| GitHub CLI | Logged in as Swathi-A01 |
| Python | 3.11 available |

**Decision:** Use Docker Desktop Kubernetes for Task 7 (already installed, just needs enabling).
No Minikube needed. No Ubuntu needed — everything runs on macOS.

---

## Why No Ubuntu?

The lecturer used Ubuntu in Class 2 specifically for Apache Airflow (which has Linux-first support).
This assignment does not use Airflow. Python, Docker, FastAPI, Kubernetes, GitHub Actions all
run identically on macOS.

---

## Git Commit

```
Initial project scaffold — full MLOps pipeline structure

Includes: data download script, preprocessing pipeline, training script
with MLflow tracking (LR + RF + XGBoost), FastAPI serving endpoint,
Pytest unit tests, GitHub Actions CI, Dockerfile, K8s manifests,
and Prometheus+Grafana monitoring config.
```
