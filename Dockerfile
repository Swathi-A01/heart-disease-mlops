# Dockerfile
# ───────────
# Builds a self-contained Docker image for the Heart Disease Risk API.
#
# Key design choice: the model is TRAINED INSIDE THE CONTAINER at build time.
# This avoids a real bug: if a .pkl file trained on one sklearn version is loaded
# by a slightly different sklearn version in the container, it crashes with
# AttributeError. Building inside the container guarantees the saved model and
# the sklearn that loads it are always identical.
#
# Image size optimisation:
#   - python:3.11-slim base (no dev tools, no docs — ~50% smaller than full image)
#   - --no-cache-dir prevents pip from storing download cache in the image layer
#   - .dockerignore excludes venv/, notebooks/, screenshots/, etc.
#
# Build:  docker build -t heart-risk-api:latest .
# Run:    docker run -d -p 8000:8000 heart-risk-api:latest
# Test:   curl http://localhost:8000/health

# Base image: slim Python 3.11 on Linux
# Using a specific tag (not :latest) ensures reproducible builds
FROM python:3.11-slim

# Set the working directory inside the container
# All subsequent COPY, RUN, CMD commands are relative to /app
WORKDIR /app

# Copy requirements first — Docker caches this layer separately.
# If requirements.txt hasn't changed, pip install is skipped on rebuild.
# This significantly speeds up repeated builds during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, data download script, and API
# Ordering matters: copy files that change less frequently first
COPY src/ src/
COPY data/ data/
COPY api/ api/

# Download the dataset inside the container
# This ensures the model trains on the exact same data regardless of host machine
RUN python data/download_data.py

# Train the model inside the container using quick-run mode
# --quick-run skips hyperparameter tuning (fast build)
# mkdir -p creates the output directories if they don't exist
RUN mkdir -p models plots && python src/train.py --quick-run

# Tell Docker this container listens on port 8000
# This is documentation — it doesn't actually publish the port
# Use -p 8000:8000 in docker run to expose it to the host
EXPOSE 8000

# HEALTHCHECK: Docker periodically runs this command to verify the container is healthy.
# If it fails 3 times in a row, the container is marked "unhealthy".
# Kubernetes can use this to restart unhealthy pods automatically.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the FastAPI server when the container starts
# --host 0.0.0.0: accept connections from outside the container (not just localhost)
# --port 8000: match the EXPOSE port above
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
