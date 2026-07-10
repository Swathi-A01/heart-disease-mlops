FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, data, and training script
COPY src/ src/
COPY data/ data/
COPY api/ api/

# Train model inside the container so sklearn versions match at inference
RUN mkdir -p models plots && python src/train.py --quick-run

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
