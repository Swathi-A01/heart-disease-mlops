# Step 08 — Monitoring with Prometheus + Grafana

**Date:** 2026-07-10
**Assignment Task:** Task 8 — Monitoring & Logging [3 marks]
**Status:** Complete

---

## What Was Done

### Architecture

```
FastAPI (/metrics endpoint)
    ↑ scrapes every 15s
Prometheus (localhost:9090)
    ↑ data source
Grafana (localhost:3000)
```

### How /metrics is exposed in the API

One line in `api/main.py` does everything:
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```
This auto-tracks: request count, latency histogram, status codes — per endpoint.

Custom metric added for business-level monitoring:
```python
prediction_counter = Counter("heart_risk_predictions_total", ..., ["risk_level"])
# incremented on every /predict call
```

### Start Monitoring Stack

```bash
cd monitoring/
docker compose up -d
```

Starts:
- `prometheus` on port 9090 — scrapes API at `host.docker.internal:80/metrics`
- `grafana` on port 3000 — visualises Prometheus data

### Grafana Setup

1. Open http://localhost:3000 (admin/admin)
2. Connections → Data sources → Add → Prometheus → URL: http://prometheus:9090
3. Dashboards → Import → Upload `monitoring/grafana-dashboard.json`

### Dashboard Panels

| Panel | PromQL Query | What it shows |
|-------|-------------|---------------|
| Request rate | `rate(http_requests_total{handler="/predict"}[1m])` | Requests/sec to /predict |
| Predictions by risk | `heart_risk_predictions_total` | High vs low risk over time |
| Latency p50/p95 | `histogram_quantile(0.95, ...)` | How fast the API responds |
| Success rate | `rate(http_requests_total{status="2xx"}[5m])` | API reliability |

### Prometheus Target Verification

```
Job:    heart-risk-api
URL:    http://host.docker.internal:80/metrics
Health: up
```

### Metrics Confirmed Live

```
heart_risk_predictions_total{risk_level="high"} 6.0
heart_risk_predictions_total{risk_level="low"}  4.0
http_requests_total{handler="/predict",status="2xx"} 10.0
```

---

## Why Monitoring Matters (for the report)

Without monitoring you have no visibility into:
- Is the API up? (uptime)
- Is it getting slower over time? (latency drift)
- Are predictions skewed — suddenly all "high risk"? (data/model drift signal)
- Is there a spike in errors? (bug in production)

Prometheus + Grafana is the standard OSS stack used in production ML systems.
