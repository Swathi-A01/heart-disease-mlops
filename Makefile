.PHONY: install test lint format train serve evaluate predict stack-up stack-down k8s-deploy k8s-delete clean

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	@echo "Dependencies installed."

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	python -m flake8 src/ api/ tests/ --max-line-length=100 --ignore=E402
	@echo "Lint passed."

test:
	pytest tests/ -v --tb=short --junitxml=test-results.xml

# ── Data & Training ───────────────────────────────────────────────────────────

data:
	python data/download_data.py

train: data
	python src/train.py
	@echo "Training complete. Model saved to models/pipeline.pkl"

train-quick:
	python src/train.py --quick-run

evaluate:
	python src/evaluate.py

# ── Serving ───────────────────────────────────────────────────────────────────

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ── Standalone inference ──────────────────────────────────────────────────────

predict-sample:
	python src/predict.py \
		--age 67 --sex 1 --cp 4 --trestbps 160 --chol 286 \
		--fbs 0 --restecg 2 --thalach 108 --exang 1 \
		--oldpeak 1.5 --slope 2 --ca 3 --thal 7

predict-batch:
	python src/predict.py --input data/heart.csv --output predictions.csv

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t heart-risk-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name heart-risk heart-risk-api:latest
	@echo "API running at http://localhost:8000/docs"

docker-stop:
	docker stop heart-risk && docker rm heart-risk

# ── Full stack (API + MLflow + Prometheus + Grafana) ──────────────────────────

stack-up:
	docker compose -f docker-compose.full.yml up -d
	@echo ""
	@echo "Stack running:"
	@echo "  API:        http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5001"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000  (admin/admin)"

stack-down:
	docker compose -f docker-compose.full.yml down

stack-logs:
	docker compose -f docker-compose.full.yml logs -f

# ── Kubernetes ────────────────────────────────────────────────────────────────

k8s-deploy:
	kubectl apply -f k8s/
	@echo "Waiting for pods..."
	kubectl rollout status deployment/heart-risk-deployment
	kubectl get pods
	kubectl get services

k8s-delete:
	kubectl delete -f k8s/

k8s-logs:
	kubectl logs deployment/heart-risk-deployment --tail=50

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f test-results.xml predictions.csv
	@echo "Cleaned."
