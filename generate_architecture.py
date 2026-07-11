"""
Clean professional architecture diagram — white background, Lucidchart style.
Fixed spacing, no overlaps.
Saves to screenshots/architecture_clean.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("screenshots/architecture_clean.png")

fig, ax = plt.subplots(figsize=(22, 16))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")

# ── colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "data":    ("#DBEAFE", "#1D4ED8"),
    "feat":    ("#EDE9FE", "#6D28D9"),
    "train":   ("#DCFCE7", "#15803D"),
    "mlflow":  ("#FEF9C3", "#A16207"),
    "api":     ("#FFE4E6", "#BE123C"),
    "docker":  ("#CCFBF1", "#0F766E"),
    "k8s":     ("#DBEAFE", "#1E40AF"),
    "render":  ("#DCFCE7", "#166534"),
    "ci":      ("#FEF3C7", "#B45309"),
    "monitor": ("#FCE7F3", "#9D174D"),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def box(x, y, w, h, key, title, lines=None, radius=0.2):
    fill, border = COLORS[key]
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=2, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(p)
    # title bar
    bar = FancyBboxPatch((x, y+h-0.52), w, 0.52,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=border, alpha=0.85, zorder=4)
    ax.add_patch(bar)
    ax.text(x + w/2, y + h - 0.26, title,
        fontsize=9, fontweight="bold", ha="center", va="center",
        color="white", zorder=5)
    if lines:
        total = len(lines)
        for i, line in enumerate(lines):
            ly = y + h - 0.52 - 0.38 * (i + 1)
            ax.text(x + w/2, ly, line,
                fontsize=7.8, ha="center", va="center",
                color="#1a1a1a", zorder=5)

def chip(x, y, text, key, size=7.2):
    fill, border = COLORS[key]
    w = len(text) * 0.087 + 0.22
    p = FancyBboxPatch((x - w/2, y - 0.14), w, 0.28,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=1.2, edgecolor=border, facecolor=fill, zorder=5)
    ax.add_patch(p)
    ax.text(x, y, text, fontsize=size, ha="center", va="center",
        color=border, fontweight="bold", zorder=6)

def arr(x1, y1, x2, y2, key, lbl="", rad=0.0):
    _, color = COLORS[key]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
        mutation_scale=13, connectionstyle=f"arc3,rad={rad}"), zorder=4)
    if lbl:
        mx, my = (x1+x2)/2 + 0.05, (y1+y2)/2
        ax.text(mx, my, lbl, fontsize=6.5, color=color,
            style="italic", ha="left", va="center", zorder=6)

def stat_chip(x, y, text, key):
    _, border = COLORS[key]
    w = len(text) * 0.092 + 0.28
    p = FancyBboxPatch((x, y - 0.2), w, 0.4,
        boxstyle="round,pad=0,rounding_size=0.1",
        linewidth=1.5, edgecolor=border, facecolor="white", zorder=4)
    ax.add_patch(p)
    ax.text(x + w/2, y, text, fontsize=8, ha="center", va="center",
        color=border, fontweight="bold", zorder=5)
    return x + w + 0.22

# ═══════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════
ax.text(11, 15.65, "Heart Disease Risk Prediction — MLOps System Architecture",
    fontsize=18, fontweight="bold", ha="center", color="#111827")
ax.text(11, 15.2, "UCI Dataset  ·  scikit-learn + XGBoost  ·  MLflow  ·  FastAPI  ·  Docker  ·  Kubernetes  ·  Render  ·  Prometheus + Grafana  ·  GitHub Actions",
    fontsize=9, ha="center", color="#6B7280", style="italic")
ax.plot([0.5, 21.5], [14.9, 14.9], color="#E5E7EB", lw=1.5)

# ═══════════════════════════════════════════════════════════════════
# ROW 1  — Data · Feature Eng · Training · MLflow
# ═══════════════════════════════════════════════════════════════════
ROW1_Y = 11.5
BOX_H  = 2.9

box(0.4,  ROW1_Y, 3.8, BOX_H, "data",   "DATA LAYER",
    ["UCI Heart Disease (Cleveland)", "303 patients · 14 features", "download_data.py", "na_values='?'  ·  binarise target"])

box(4.9,  ROW1_Y, 4.2, BOX_H, "feat",   "FEATURE ENGINEERING",
    ["StandardScaler (9 numeric)", "OneHotEncoder drop='first' (6 cat)", "Passthrough (3 binary)", "5 derived: hr_reserve · bp_cat · chol_risk"])

box(9.8,  ROW1_Y, 4.3, BOX_H, "train",  "MODEL TRAINING",
    ["Logistic Regression  (GridSearchCV)", "Random Forest  (RandomizedSearchCV)", "XGBoost  (RandomizedSearchCV)", "StratifiedKFold 5-fold · scoring=roc_auc"])

box(14.8, ROW1_Y, 4.2, BOX_H, "mlflow", "MLFLOW TRACKING",
    ["params: model · hyperparameters", "metrics: roc_auc · cv_roc_auc · f1", "artifacts: CM · ROC · PR · calibration", "sqlite:///mlflow.db  ·  localhost:5000"])

# Row 1 horizontal arrows
arr(4.2,  ROW1_Y + 1.45, 4.9,  ROW1_Y + 1.45, "data")
arr(9.1,  ROW1_Y + 1.45, 9.8,  ROW1_Y + 1.45, "feat")
arr(14.1, ROW1_Y + 1.45, 14.8, ROW1_Y + 1.45, "train", "log all runs")

# Best model arrow downward
ax.text(11.95, ROW1_Y - 0.22, "★  Best model  →  models/pipeline.pkl",
    fontsize=8, ha="center", color="#15803D", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#DCFCE7",
              edgecolor="#15803D", linewidth=1))
arr(11.95, ROW1_Y, 11.95, ROW1_Y - 0.18, "train")

# ═══════════════════════════════════════════════════════════════════
# ROW 2  — FastAPI · Docker · Kubernetes · Render
# ═══════════════════════════════════════════════════════════════════
ROW2_Y = 7.8

box(0.4,  ROW2_Y, 4.8, BOX_H, "api",    "FASTAPI  (api/main.py)",
    ["POST /predict  — single patient", "POST /predict-batch  — up to 100", "GET /health · /ready · /model-info", "GET /stats · /metrics (Prometheus)"])

box(5.9,  ROW2_Y, 4.0, BOX_H, "docker", "DOCKER",
    ["FROM python:3.11-slim", "RUN download_data.py + train.py", "HEALTHCHECK GET /health", "CMD uvicorn api.main:app :8000"])

box(10.6, ROW2_Y, 4.2, BOX_H, "k8s",   "KUBERNETES  (Docker Desktop)",
    ["Deployment: 2 replicas", "Readiness probe: GET /ready", "Liveness probe: GET /health", "Service: LoadBalancer  port 80→8000"])

box(15.5, ROW2_Y, 4.2, BOX_H, "render","RENDER CLOUD",
    ["Dockerfile.render  (lightweight)", "Auto-deploy on git push to main", "HTTPS · Singapore region · Free tier",
     "heart-disease-mlops-rcg4.onrender.com"])

# Row 2 arrows
arr(5.2,  ROW2_Y + 1.45, 5.9,  ROW2_Y + 1.45, "api",    "containerise")
arr(9.9,  ROW2_Y + 1.45, 10.6, ROW2_Y + 1.45, "docker", "deploy")
arr(14.8, ROW2_Y + 1.45, 15.5, ROW2_Y + 1.45, "k8s",    "cloud")

# pipeline.pkl → FastAPI
arr(11.95, ROW1_Y - 0.38, 2.8, ROW2_Y + BOX_H,
    "train", "pipeline.pkl", rad=-0.25)

# ═══════════════════════════════════════════════════════════════════
# ROW 3  — CI/CD  |  Monitoring
# ═══════════════════════════════════════════════════════════════════
ROW3_Y = 4.2

box(0.4, ROW3_Y, 10.0, 3.0, "ci",
    "CI/CD  —  GitHub Actions  (.github/workflows/ci.yml)",
    ["push to main  →  ubuntu-latest VM  →  fail fast on errors"])

# CI step boxes
ci_steps = ["flake8\nlint", "download\ndataset", "pytest\n10 tests", "train\n--quick-run", "pytest API\n15 tests", "upload\nartifacts"]
for i, s in enumerate(ci_steps):
    bx = 0.7 + i * 1.6
    inner = FancyBboxPatch((bx, ROW3_Y + 0.4), 1.35, 1.85,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=1.2, edgecolor="#B45309", facecolor="#FFFBEB", zorder=4)
    ax.add_patch(inner)
    ax.text(bx + 0.675, ROW3_Y + 1.325, s, fontsize=8, ha="center",
        va="center", color="#92400E", fontweight="bold", zorder=5)
    if i < len(ci_steps) - 1:
        arr(bx + 1.35, ROW3_Y + 1.325, bx + 1.6, ROW3_Y + 1.325, "ci")

box(11.1, ROW3_Y, 10.5, 3.0, "monitor",
    "MONITORING  —  Prometheus  +  Grafana  (docker-compose)",
    ["API exposes /metrics  ·  scrape every 15s  ·  13-panel Grafana dashboard"])

mon = [
    ("Prometheus\n:9090", "scrapes\n/metrics"),
    ("Grafana\n:3000", "13 panels\ndashboard"),
    ("Counter\npredictions", "high/low\nrisk split"),
    ("Histogram\nlatency", "p50·p95·p99\nper call"),
    ("Structured\nLogs", "age·cp·prob\nper request"),
]
for i, (top, bot) in enumerate(mon):
    bx = 11.4 + i * 2.0
    inner = FancyBboxPatch((bx, ROW3_Y + 0.4), 1.75, 1.85,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=1.2, edgecolor="#9D174D", facecolor="#FDF2F8", zorder=4)
    ax.add_patch(inner)
    ax.text(bx + 0.875, ROW3_Y + 1.55, top, fontsize=8, ha="center",
        va="center", color="#9D174D", fontweight="bold", zorder=5)
    ax.text(bx + 0.875, ROW3_Y + 0.82, bot, fontsize=7, ha="center",
        va="center", color="#6B7280", zorder=5)

# API → monitoring arrow
arr(2.8, ROW2_Y, 2.8, ROW3_Y + 3.0, "monitor", "/metrics", rad=0)

# ═══════════════════════════════════════════════════════════════════
# BOTTOM BAR
# ═══════════════════════════════════════════════════════════════════
ax.plot([0.5, 21.5], [3.9, 3.9], color="#E5E7EB", lw=1.5)

ax.text(1.6, 3.5, "GitHub Repository",
    fontsize=9, fontweight="bold", color="#111827", ha="center")
ax.text(1.6, 3.15, "github.com/Swathi-A01/\nheart-disease-mlops",
    fontsize=8, color="#1D4ED8", ha="center")

# stat chips
chips = [
    ("297 patients",        "data"),
    ("18 features total",   "feat"),
    ("3 models trained",    "train"),
    ("ROC-AUC  0.9464",     "train"),
    ("25 pytest tests",     "api"),
    ("11 CI steps",         "ci"),
    ("2 K8s replicas",      "k8s"),
    ("13 Grafana panels",   "monitor"),
]
cx = 3.3
for txt, key in chips:
    cx = stat_chip(cx, 3.3, txt, key)

# Live URL
url_box = FancyBboxPatch((3.5, 2.55), 15.0, 0.6,
    boxstyle="round,pad=0,rounding_size=0.15",
    linewidth=1.8, edgecolor="#166534", facecolor="#F0FDF4", zorder=3)
ax.add_patch(url_box)
ax.text(11.0, 2.85,
    "Live API:   https://heart-disease-mlops-rcg4.onrender.com/docs",
    fontsize=10, ha="center", va="center",
    color="#166534", fontweight="bold", zorder=4)

# outer border
outer = FancyBboxPatch((0.2, 2.35), 21.6, 13.2,
    boxstyle="round,pad=0,rounding_size=0.3",
    linewidth=2, edgecolor="#D1D5DB", facecolor="none", zorder=1)
ax.add_patch(outer)

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=180, bbox_inches="tight",
    facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
