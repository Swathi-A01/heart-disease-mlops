"""
Generate a professional system architecture diagram — Figma/LucidChart style.
Dark theme, color-coded layers, clean typography, styled arrows.
Saves to plots/architecture_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path

OUT = Path("plots/architecture_diagram.png")

# ── colour palette ────────────────────────────────────────────────────────────
BG       = "#0D1117"
C_DATA   = "#58A6FF"   # blue
C_FEAT   = "#BC8CFF"   # purple
C_TRAIN  = "#3FB950"   # green
C_API    = "#F78166"   # coral / orange
C_DOCKER = "#56D364"   # bright green
C_K8S    = "#388BFD"   # sky blue
C_MON    = "#FF7B72"   # red-coral
C_CI     = "#D29922"   # amber/gold
C_TEXT   = "#E6EDF3"
C_MUTED  = "#8B949E"
FONT     = "Avenir"

fig, ax = plt.subplots(figsize=(22, 15))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 15)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────

def card(x, y, w, h, color, alpha_fill=0.10):
    """Rounded card with glowing border."""
    # glow
    glow = FancyBboxPatch((x-0.05, y-0.05), w+0.1, h+0.1,
        boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=4, edgecolor=color, facecolor="none", alpha=0.15, zorder=2)
    ax.add_patch(glow)
    # fill
    fill = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.25",
        linewidth=0, facecolor=color, alpha=alpha_fill, zorder=3)
    ax.add_patch(fill)
    # border
    border = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.25",
        linewidth=1.2, edgecolor=color, facecolor="none", alpha=0.8, zorder=4)
    ax.add_patch(border)


def header(x, y, w, text, color):
    """Section header bar."""
    h_fill = FancyBboxPatch((x, y), w, 0.5,
        boxstyle="round,pad=0,rounding_size=0.2",
        linewidth=0, facecolor=color, alpha=0.9, zorder=4)
    ax.add_patch(h_fill)
    ax.text(x + w/2, y + 0.25, text, fontsize=10, color="white",
            ha="center", va="center", fontweight="bold",
            fontfamily=FONT, zorder=5)


def chip(x, y, w, h, color, text_top, text_bot="", alpha=0.22):
    """Small chip/pill inside a section."""
    fill = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.15",
        linewidth=1, edgecolor=color, facecolor=color, alpha=alpha, zorder=5)
    ax.add_patch(fill)
    mid = y + h / 2
    if text_bot:
        ax.text(x + w/2, mid + 0.13, text_top, fontsize=7.5, color=C_TEXT,
                ha="center", va="center", fontweight="bold",
                fontfamily=FONT, zorder=6)
        ax.text(x + w/2, mid - 0.15, text_bot, fontsize=6.5, color=C_MUTED,
                ha="center", va="center", fontfamily=FONT, zorder=6)
    else:
        ax.text(x + w/2, mid, text_top, fontsize=7.5, color=C_TEXT,
                ha="center", va="center", fontweight="bold",
                fontfamily=FONT, zorder=6)


def txt(x, y, s, size=8.5, color=C_TEXT, bold=False, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            fontfamily=FONT, zorder=6)


def dot_arrow(x1, y1, x2, y2, color, lw=2.0, rad=0.0, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=14,
        ), zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=6.5, color=color, style="italic",
                ha="left", va="center", fontfamily=FONT, zorder=7)


def badge(x, y, text, color):
    """Small colored badge for method labels."""
    w = len(text) * 0.085 + 0.15
    fill = FancyBboxPatch((x, y-0.12), w, 0.25,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=0, facecolor=color, alpha=0.85, zorder=6)
    ax.add_patch(fill)
    ax.text(x + w/2, y+0.005, text, fontsize=6, color="white",
            ha="center", va="center", fontweight="bold",
            fontfamily=FONT, zorder=7)
    return w


# ════════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════════

ax.text(11, 14.5, "Heart Disease Risk Prediction", fontsize=22,
        color=C_TEXT, ha="center", va="center", fontweight="bold", fontfamily=FONT)
ax.text(11, 14.0, "End-to-End MLOps System Architecture",
        fontsize=11, color=C_MUTED, ha="center", va="center", fontfamily=FONT)

# thin divider line
ax.plot([1, 21], [13.7, 13.7], color="#30363D", lw=1, zorder=2)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 1  ·  DATA (left)  +  CI/CD (right)
# ════════════════════════════════════════════════════════════════════════════════

card(0.4, 11.0, 6.5, 2.5, C_DATA)
header(0.4, 13.0, 6.5, "DATA LAYER", C_DATA)
chip(0.65, 11.2, 1.6, 1.55, C_DATA, "UCI CSV", "303 rows · 14 cols")
chip(2.45, 11.2, 1.9, 1.55, C_DATA, "download_data.py", "headers · NaN · binarise")
chip(4.55, 11.2, 1.95, 1.55, C_DATA, "heart.csv", "297 rows · clean")
dot_arrow(2.25, 11.975, 2.45, 11.975, C_DATA)
dot_arrow(4.35, 11.975, 4.55, 11.975, C_DATA)

card(14.8, 11.0, 6.8, 2.5, C_CI)
header(14.8, 13.0, 6.8, "CI/CD  —  GitHub Actions", C_CI)
ci_steps = [("flake8", "lint"), ("pytest", "24 tests"), ("train.py", "--quick-run"), ("artifacts", "upload")]
for i, (a, b) in enumerate(ci_steps):
    xi = 15.1 + i * 1.6
    chip(xi, 11.2, 1.35, 1.55, C_CI, a, b)
    if i < 3:
        dot_arrow(xi + 1.35, 11.975, xi + 1.6, 11.975, C_CI)
txt(18.2, 11.1, "push > auto-trigger > fail on errors", size=7.5, color=C_MUTED)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 2  ·  FEATURE ENG  +  TRAINING  +  MLFLOW
# ════════════════════════════════════════════════════════════════════════════════

card(0.4, 8.3, 6.5, 2.45, C_FEAT)
header(0.4, 10.3, 6.5, "FEATURE ENGINEERING  (preprocess.py)", C_FEAT)
fe_items = [
    ("StandardScaler", "6 numeric"),
    ("OneHotEncoder", "6 categorical"),
    ("Passthrough", "3 binary"),
    ("Derived", "hr_reserve\nbp_category\nchol_risk"),
]
for i, (a, b) in enumerate(fe_items):
    xi = 0.65 + i * 1.55
    chip(xi, 8.5, 1.3, 1.6, C_FEAT, a, b)
txt(3.65, 8.4, "ColumnTransformer · fitted on train set only · no leakage", size=7.5, color=C_MUTED)

card(7.2, 8.3, 5.2, 2.45, C_TRAIN)
header(7.2, 10.3, 5.2, "MODEL TRAINING  (train.py)", C_TRAIN)
models = [("LogisticReg", "GridSearchCV\nC=[0.01..100]"),
          ("Random\nForest", "RandomizedCV\n10 iterations"),
          ("XGBoost", "RandomizedCV\n8 iterations")]
for i, (a, b) in enumerate(models):
    xi = 7.45 + i * 1.6
    chip(xi, 8.5, 1.4, 1.6, C_TRAIN, a, b)
txt(9.8, 8.4, "StratifiedKFold · ROC-AUC · best model > pipeline.pkl", size=7.5, color=C_MUTED)

card(12.7, 8.3, 5.0, 2.45, C_TRAIN, alpha_fill=0.07)
header(12.7, 10.3, 5.0, "MLFLOW TRACKING", C_TRAIN)
ml_items = [("params", "C, n_estimators\nmax_depth"), ("metrics", "roc_auc\ncv_roc_auc·f1"),
            ("artifacts", "ROC·CM\nPR·calib·pkl")]
for i, (a, b) in enumerate(ml_items):
    xi = 12.95 + i * 1.55
    chip(xi, 8.5, 1.35, 1.6, C_TRAIN, a, b)
txt(15.2, 8.4, "sqlite:///mlflow.db · 3 experiments logged", size=7.5, color=C_MUTED)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 3  ·  FASTAPI  +  DOCKER
# ════════════════════════════════════════════════════════════════════════════════

card(0.4, 5.5, 7.8, 2.5, C_API)
header(0.4, 7.5, 7.8, "FASTAPI  (api/main.py)  —  uvicorn port 8000", C_API)

endpoints = [
    ("POST", "/predict", "heart_rate_reserve\n+ age_thalach_ratio"),
    ("POST", "/predict-batch", "up to 100 patients\naggregate summary"),
    ("GET",  "/model-info",    "model type\n+ feature list"),
    ("GET",  "/stats",         "live prediction\ncounters"),
    ("GET",  "/health",        "liveness\ncheck"),
    ("GET",  "/metrics",       "Prometheus\ncounters"),
]
get_col = "#3FB950"; post_col = "#F78166"
for i, (method, ep, desc) in enumerate(endpoints):
    col_i = i % 3; row_i = i // 3
    xb = 0.65 + col_i * 2.55
    yb = 6.9 - row_i * 1.2
    mc = get_col if method == "GET" else post_col
    chip(xb, yb - 0.85, 2.25, 1.1, C_API, ep, desc, alpha=0.18)
    bw = badge(xb + 0.06, yb - 0.1, method, mc)

txt(4.3, 5.65, "Pydantic validation · structured logging · engineer_features() at inference", size=7.5, color=C_MUTED)

card(8.5, 5.5, 6.6, 2.5, C_DOCKER)
header(8.5, 7.5, 6.6, "DOCKER  (python:3.11-slim)", C_DOCKER)
dk_steps = [("FROM\npython:3.11", "base image"),
            ("pip install\nrequirements", "16 deps"),
            ("download_data\n+ train.py", "build-time"),
            ("uvicorn\n:8000", "HEALTHCHECK")]
for i, (a, b) in enumerate(dk_steps):
    xi = 8.75 + i * 1.55
    chip(xi, 5.7, 1.35, 1.6, C_DOCKER, a, b)
    if i < 3:
        dot_arrow(xi + 1.35, 6.5, xi + 1.55, 6.5, C_DOCKER)
txt(11.8, 5.65, ".dockerignore · model trained at build time · no version mismatch", size=7.5, color=C_MUTED)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 4  ·  KUBERNETES  +  MONITORING
# ════════════════════════════════════════════════════════════════════════════════

card(0.4, 2.8, 7.8, 2.4, C_K8S)
header(0.4, 4.7, 7.8, "KUBERNETES  —  Docker Desktop", C_K8S)
k8s_items = [("Deployment", "2 replicas\nimagePullPolicy\nIfNotPresent"),
             ("Service\nLoadBalancer", "port 80\n> 8000"),
             ("Readiness\nProbe", "/health\nevery 5s"),
             ("Liveness\nProbe", "/health\nevery 10s"),
             ("Resources", "256Mi / 500m\nreq · limits")]
for i, (a, b) in enumerate(k8s_items):
    xi = 0.65 + i * 1.5
    chip(xi, 3.0, 1.3, 1.5, C_K8S, a, b)
txt(4.3, 2.95, "kubectl apply -f k8s/ · 2 pods Running · curl http://localhost/predict", size=7.5, color=C_MUTED)

card(8.5, 2.8, 6.6, 2.4, C_MON)
header(8.5, 4.7, 6.6, "MONITORING  (docker-compose)", C_MON)
mon_items = [("Prometheus\n:9090", "scrape /metrics\nevery 15s"),
             ("Grafana\n:3000", "4-panel\ndashboard"),
             ("Custom\nCounters", "predictions_total\nby risk_level"),
             ("Latency\nHistogram", "per-prediction\ntiming")]
for i, (a, b) in enumerate(mon_items):
    xi = 8.75 + i * 1.55
    chip(xi, 3.0, 1.35, 1.5, C_MON, a, b)
txt(11.8, 2.95, "Prometheus > Grafana · structured logs · /stats endpoint", size=7.5, color=C_MUTED)

# ════════════════════════════════════════════════════════════════════════════════
# BOTTOM  ·  REPO
# ════════════════════════════════════════════════════════════════════════════════

card(2.5, 0.4, 10.5, 2.1, C_DATA, alpha_fill=0.07)
header(2.5, 2.0, 10.5, "github.com/Swathi-A01/heart-disease-mlops", C_DATA)
repo_items = ["data/", "src/", "api/", "tests/\n24 tests", "k8s/", "monitoring/", ".github/\nworkflows"]
for i, ri in enumerate(repo_items):
    xi = 2.75 + i * 1.45
    chip(xi, 0.6, 1.25, 1.2, C_DATA, ri, alpha=0.2)

# ════════════════════════════════════════════════════════════════════════════════
# ARROWS — between rows
# ════════════════════════════════════════════════════════════════════════════════

# Data > FE
dot_arrow(3.65, 11.0, 3.65, 10.78, C_DATA, lw=2)

# FE > Training
dot_arrow(6.9, 9.52, 7.2, 9.52, C_FEAT, lw=2)

# Training > MLflow
dot_arrow(12.4, 9.52, 12.7, 9.52, C_TRAIN, lw=2)

# Training > FastAPI (pipeline.pkl)
dot_arrow(9.8, 8.3, 4.0, 8.02, C_TRAIN, lw=2, rad=-0.2, label="pipeline.pkl")

# FastAPI > Docker
dot_arrow(8.2, 6.75, 8.5, 6.75, C_API, lw=2)

# Docker > K8s
dot_arrow(11.8, 5.5, 5.5, 5.22, C_DOCKER, lw=2, rad=0.15, label="docker image")

# K8s > Monitoring (/metrics)
dot_arrow(8.2, 3.8, 8.5, 3.8, C_K8S, lw=2, label="/metrics")

# CI/CD downward
dot_arrow(18.2, 11.0, 18.2, 9.8, C_CI, lw=2, label="push trigger")

# Repo ↑
dot_arrow(7.75, 2.5, 7.75, 2.8, C_DATA, lw=1.5)

# ════════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════════

legend_items = [(C_DATA,"Data"), (C_FEAT,"Feature Eng."), (C_TRAIN,"Training / MLflow"),
                (C_API,"FastAPI"), (C_DOCKER,"Docker"), (C_K8S,"Kubernetes"),
                (C_MON,"Monitoring"), (C_CI,"CI/CD")]

lx, ly = 15.5, 4.6
txt(17.0, ly + 0.3, "LEGEND", size=8, color=C_MUTED, bold=True)
for i, (col, name) in enumerate(legend_items):
    xi = lx + (i % 4) * 1.65
    yi = ly - 0.45 - (i // 4) * 0.5
    c = Circle((xi + 0.1, yi + 0.1), 0.09, color=col, zorder=6)
    ax.add_patch(c)
    ax.text(xi + 0.25, yi + 0.1, name, fontsize=7.5, color=C_TEXT,
            va="center", fontfamily=FONT, zorder=6)

# ════════════════════════════════════════════════════════════════════════════════

plt.tight_layout(pad=0.1)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
