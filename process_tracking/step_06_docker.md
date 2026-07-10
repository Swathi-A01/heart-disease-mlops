# Step 06 — Docker Containerization

**Date:** 2026-07-10
**Assignment Task:** Task 6 — Model Containerization [5 marks]
**Status:** Complete

---

## What Was Done

### Dockerfile

```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt → pip install
COPY src/ data/ api/
RUN python src/train.py --quick-run   ← trains model INSIDE container
EXPOSE 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Build and Test Commands

```bash
docker build -t heart-risk-api:latest .
docker run -d -p 8000:8000 --name heart-risk-test heart-risk-api:latest

# Health check
curl http://localhost:8000/health
# → {"status":"ok","model":"heart-disease-classifier"}

# Low risk patient
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":1,...,"thal":3}'
# → {"prediction":0,"probability":0.1056,"risk":"low"}

# High risk patient
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,...,"thal":7}'
# → {"prediction":1,"probability":0.9982,"risk":"high"}
```

---

## Issues Encountered & Fixed

### sklearn version mismatch (AttributeError: 'LogisticRegression' has no attribute 'multi_class')

**Root cause:** `pipeline.pkl` was saved by sklearn 1.4.0 on the Mac. The Docker container
pulled a newer patch version (1.4.x → 1.5.x internally in python:3.11-slim) where the
`multi_class` attribute was removed from LogisticRegression.

**Fix:** Train the model **inside** the Docker container at build time using
`RUN python src/train.py --quick-run`. This guarantees the saved `.pkl` and the
sklearn version that loads it are identical — no cross-version compatibility issues possible.

**Why this is the right MLOps pattern:** The model and its runtime environment should
always be packaged together. Copying a `.pkl` from outside the container is fragile.
Building it inside the container is reproducible.

---

## Key Design Decisions

- `python:3.11-slim` base — ~50% smaller than `python:3.11` (no dev tools, no docs)
- Model trained at build time (`RUN`) not at startup (`CMD`) — startup is instant
- `--quick-run` flag used in Docker — no hyperparameter tuning, just verifies pipeline works
- Port 8000 exposed — matches uvicorn default, easy to map with `-p 8000:8000`
- Container is stateless — no volumes, no persistent DB needed for serving
