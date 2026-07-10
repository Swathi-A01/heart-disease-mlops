# Heart Disease MLOps Pipeline

End-to-end MLOps project predicting heart disease risk using the UCI Heart Disease dataset.

## Tasks Covered
- EDA with professional visualizations
- ML model training (Logistic Regression, Random Forest, XGBoost)
- Experiment tracking with MLflow
- FastAPI serving endpoint
- Docker containerization
- Kubernetes deployment (Minikube)
- CI/CD with GitHub Actions
- Monitoring with Prometheus + Grafana

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Swathi-A01/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data
python data/download_data.py

# 3. Train models
python src/train.py

# 4. Run API
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs

# 5. Run tests
pytest tests/ -v
```

## Project Structure

```
├── data/           # Dataset + download script
├── notebooks/      # EDA notebook
├── src/            # Preprocessing + training code
├── api/            # FastAPI application
├── tests/          # Pytest unit tests
├── k8s/            # Kubernetes manifests
├── monitoring/     # Prometheus + Grafana config
└── .github/        # GitHub Actions CI/CD
```
