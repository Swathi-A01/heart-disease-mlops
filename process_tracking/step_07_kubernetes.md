# Step 07 — Kubernetes Deployment

**Date:** 2026-07-10
**Assignment Task:** Task 7 — Production Deployment [7 marks]
**Status:** Complete

---

## What Was Done

Deployed the Docker container to **Docker Desktop Kubernetes** (local K8s — no cloud cost).

### Manifests Applied

**`k8s/deployment.yaml`**
- 2 replicas (high availability — if one pod crashes, the other keeps serving)
- Resource limits: 256Mi/500m requests, 512Mi/500m CPU limits
- `imagePullPolicy: IfNotPresent` — uses local image, no Docker Hub needed

**`k8s/service.yaml`**
- Type: LoadBalancer — exposes the service externally
- Port 80 (external) → 8000 (container)
- Docker Desktop maps LoadBalancer to `localhost` automatically

### Commands Run

```bash
# Apply both manifests
kubectl apply -f k8s/

# Check pods came up
kubectl get pods
kubectl get services

# Test endpoint
curl http://localhost/health
curl -X POST http://localhost/predict -H "Content-Type: application/json" -d '{...}'

# Check logs
kubectl logs <pod-name>
```

### Final State

```
PODS:      2/2 Running
SERVICE:   LoadBalancer, EXTERNAL-IP 172.18.0.5, port 80:32373
ENDPOINT:  http://localhost/predict  ← working
```

---

## Issues Encountered & Fixed

### ImagePullBackOff on both pods

**Root cause:** Two Docker contexts on the machine — `colima` (used normally) and
`desktop-linux` (Docker Desktop's own context). The image was built in `colima` but
Docker Desktop Kubernetes can only see images in the `desktop-linux` context.

**Diagnosed with:** `kubectl describe pod <name>` → showed "pull access denied"

**Fix:**
```bash
docker context use desktop-linux
docker build -t heart-risk-api:latest .
kubectl rollout restart deployment/heart-risk-deployment
```

**Key lesson:** Always build the image in the same Docker context that Kubernetes uses.
Check with `docker context ls` — the active context (marked `*`) must match.

---

## Key Concepts Learned

**Deployment** — tells K8s what to run, how many replicas, which image.
Always use replicas ≥ 2 in production so one pod crashing doesn't take down the service.

**Service** — tells K8s how to expose the deployment to the outside world.
LoadBalancer type on Docker Desktop automatically maps to localhost.

**`kubectl rollout restart`** — gracefully replaces all pods with fresh ones.
Used after rebuilding an image — K8s doesn't auto-detect local image changes.

**`imagePullPolicy: IfNotPresent`** — use local image if it exists, don't try Docker Hub.
Essential for local development/testing. In production you'd use a real registry (ECR, GCR, etc.)
