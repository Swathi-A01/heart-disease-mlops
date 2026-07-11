"""
Clean professional architecture diagram.
Strict grid layout — no auto-positioning, no overlaps.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from pathlib import Path

OUT = Path("screenshots/architecture_clean.png")

W, H = 28, 14
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ── Strict colour palette ──────────────────────────────────────────────────
P = {
    "blue":   ("#EBF5FF", "#2563EB", "#1D4ED8"),
    "purple": ("#F5F0FF", "#7C3AED", "#6D28D9"),
    "green":  ("#ECFDF5", "#059669", "#047857"),
    "amber":  ("#FFFBEB", "#D97706", "#B45309"),
    "red":    ("#FFF1F2", "#E11D48", "#BE123C"),
    "teal":   ("#F0FDFA", "#0D9488", "#0F766E"),
    "indigo": ("#EEF2FF", "#4F46E5", "#4338CA"),
    "emerald":("#F0FDF4", "#10B981", "#059669"),
    "orange": ("#FFF7ED", "#EA580C", "#C2410C"),
    "pink":   ("#FDF2F8", "#DB2777", "#BE185D"),
    "gray":   ("#F9FAFB", "#6B7280", "#374151"),
}

def box(x, y, w, h, color_key, title, body_lines=None):
    bg, border, dark = P[color_key]
    # shadow
    shadow = FancyBboxPatch((x+0.06, y-0.06), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0, facecolor="#DDDDDD", alpha=0.5, zorder=2)
    ax.add_patch(shadow)
    # fill
    fill = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=2, edgecolor=border, facecolor=bg, zorder=3)
    ax.add_patch(fill)
    # header bar
    bar_h = 0.5
    bar = FancyBboxPatch((x, y+h-bar_h), w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0, facecolor=border, zorder=4)
    ax.add_patch(bar)
    ax.text(x+w/2, y+h-bar_h/2, title,
        fontsize=8.5, fontweight="bold", ha="center", va="center",
        color="white", zorder=5)
    if body_lines:
        line_h = (h - bar_h - 0.15) / (len(body_lines) + 0.5)
        for i, line in enumerate(body_lines):
            ly = y + h - bar_h - 0.18 - line_h*(i+0.8)
            ax.text(x+w/2, ly, line,
                fontsize=7.2, ha="center", va="center",
                color=dark, zorder=5)

def arrow_h(x1, y, x2, color_key, label=""):
    _, c, _ = P[color_key]
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=2,
        mutation_scale=14), zorder=4)
    if label:
        ax.text((x1+x2)/2, y+0.15, label, fontsize=6.5, color=c,
            ha="center", va="bottom", style="italic")

def arrow_v(x, y1, y2, color_key, label=""):
    _, c, _ = P[color_key]
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=2,
        mutation_scale=14), zorder=4)
    if label:
        ax.text(x+0.12, (y1+y2)/2, label, fontsize=6.5, color=c,
            ha="left", va="center", style="italic")

def arrow_angled(x1, y1, x2, y2, color_key, label=""):
    _, c, _ = P[color_key]
    # draw as two orthogonal segments: right then down
    mid_x = x2
    ax.plot([x1, mid_x], [y1, y1], color=c, lw=1.8, zorder=4)
    ax.annotate("", xy=(x2, y2), xytext=(mid_x, y1),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8,
        mutation_scale=12, connectionstyle="arc3,rad=0.0"), zorder=4)
    if label:
        ax.text((x1+x2)/2, y1+0.15, label, fontsize=6.2, color=c,
            ha="center", style="italic")

def divider(x1, x2, y, color="#E5E7EB"):
    ax.plot([x1, x2], [y, y], color=color, lw=1, linestyle="--", zorder=2)

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(14, 13.55, "Heart Disease Risk Prediction",
    fontsize=20, fontweight="bold", ha="center", color="#111827")
ax.text(14, 13.1, "End-to-End MLOps System Architecture",
    fontsize=11, ha="center", color="#6B7280", style="italic")
ax.plot([0.5, 27.5], [12.8, 12.8], color="#E5E7EB", lw=1.5)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1  ·  5 pipeline stages across the top
# ═══════════════════════════════════════════════════════════════════════════
#  [DATA]  →  [FEATURE ENG]  →  [TRAINING]  →  [MLFLOW]  →  [pipeline.pkl]

BW = 3.6
BH = 2.8
BY = 9.6
GAP = 0.7
positions = [0.3, 0.3+BW+GAP, 0.3+2*(BW+GAP), 0.3+3*(BW+GAP), 0.3+4*(BW+GAP)]

box(positions[0], BY, BW, BH, "blue",   "DATA LAYER",
    ["UCI Heart Disease (Cleveland)", "303 patients · 14 features",
     "download_data.py", "handles missing values"])

box(positions[1], BY, BW, BH, "purple", "FEATURE ENGINEERING",
    ["StandardScaler (9 numeric)", "OneHotEncoder (6 categorical)",
     "Passthrough (3 binary)", "5 derived clinical features"])

box(positions[2], BY, BW, BH, "green",  "MODEL TRAINING",
    ["Logistic Regression (GridSearchCV)", "Random Forest ★ (RandomizedCV)",
     "XGBoost (RandomizedCV)", "StratifiedKFold · 5-fold · ROC-AUC"])

box(positions[3], BY, BW, BH, "amber",  "MLFLOW TRACKING",
    ["params + metrics per run", "confusion matrix · ROC curve",
     "PR curve · calibration plot", "sqlite:///mlflow.db"])

box(positions[4], BY, BW, BH, "gray",   "BEST MODEL",
    ["Random Forest", "Test ROC-AUC: 0.9464",
     "Accuracy: 83.3%", "Saved: models/pipeline.pkl"])

# Row 1 arrows
for i in range(4):
    arrow_h(positions[i]+BW, BY+BH/2, positions[i+1], "gray")

# "best model" label on last arrow — place above the arrow
ax.text(positions[3]+BW + (positions[4]-positions[3]-BW)/2,
        BY+BH/2+0.22, "best model", fontsize=6.5, color=P["green"][1],
        ha="center", style="italic", fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# VERTICAL ARROW: pipeline.pkl → FastAPI
# ═══════════════════════════════════════════════════════════════════════════
# pipeline.pkl label — to the right of the arrow so it's not clipped
arrow_v(positions[4]+BW/2, BY, BY-0.35, "gray")
ax.text(positions[4]+BW/2+0.18, BY-0.18, "pipeline.pkl",
    fontsize=7, color=P["gray"][1], ha="left", va="center",
    style="italic", fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2  ·  FastAPI  +  Docker  +  K8s  +  Render
# ═══════════════════════════════════════════════════════════════════════════
BH2  = 2.6
BY2  = 6.5
BW2  = 4.05
GAP2 = 0.9   # wider gap so arrow labels fit
pos2 = [0.3, 0.3+BW2+GAP2, 0.3+2*(BW2+GAP2), 0.3+3*(BW2+GAP2)]

box(pos2[0], BY2, BW2, BH2, "red",     "FASTAPI  (api/main.py)",
    ["POST /predict · /predict-batch",
     "GET /health · /ready · /stats",
     "GET /model-info · /metrics",
     "Pydantic validation · logging"])

box(pos2[1], BY2, BW2, BH2, "teal",    "DOCKER",
    ["FROM python:3.11-slim",
     "RUN train.py --quick-run",
     "HEALTHCHECK GET /health",
     "CMD uvicorn api.main:app :8000"])

box(pos2[2], BY2, BW2, BH2, "indigo",  "KUBERNETES",
    ["2 replicas · Docker Desktop",
     "Readiness probe: GET /ready",
     "Liveness probe: GET /health",
     "LoadBalancer  port 80→8000"])

box(pos2[3], BY2, BW2, BH2, "emerald", "RENDER CLOUD",
    ["Dockerfile.render (lightweight)",
     "Auto-deploy on git push",
     "HTTPS · Free tier · Singapore",
     "heart-disease-mlops-rcg4"])

# Row 2 arrows with white-background labels
for (x1, x2, lbl, ck) in [
    (pos2[0]+BW2, pos2[1], "containerise", "red"),
    (pos2[1]+BW2, pos2[2], "deploy local", "teal"),
    (pos2[2]+BW2, pos2[3], "cloud deploy", "indigo"),
]:
    arrow_h(x1, BY2+BH2/2, x2, ck)
    mx = (x1 + x2) / 2
    ax.text(mx, BY2+BH2/2+0.22, lbl, fontsize=7.5, color=P[ck][1],
        ha="center", va="bottom", style="italic", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor=P[ck][1], linewidth=0.8), zorder=6)

# pipeline.pkl → FastAPI (vertical into row 2)
arrow_v(pos2[0]+BW2/2, BY2+BH2, BY2+BH2+0.02, "gray")

# ═══════════════════════════════════════════════════════════════════════════
# ROW 3  ·  CI/CD  +  Monitoring (side by side)
# ═══════════════════════════════════════════════════════════════════════════
BH3 = 2.8
BY3 = 3.2

# CI/CD left half
box(0.3, BY3, 11.1, BH3, "orange", "CI/CD  —  GitHub Actions  (.github/workflows/ci.yml)",
    ["ubuntu-latest VM  ·  triggers on push to main  ·  fails on any error"])

# CI steps inside
ci = [("flake8\nlint","orange"), ("download\ndataset","orange"),
      ("pytest\n10 tests","orange"), ("train\n--quick-run","orange"),
      ("pytest API\n15 tests","orange"), ("upload\nartifacts","orange")]
for i, (label, ck) in enumerate(ci):
    bx = 0.55 + i * 1.78
    inner = FancyBboxPatch((bx, BY3+0.38), 1.52, 1.9,
        boxstyle="round,pad=0,rounding_size=0.1",
        linewidth=1.5, edgecolor=P[ck][1], facecolor="#FFFDF5", zorder=4)
    ax.add_patch(inner)
    ax.text(bx+0.76, BY3+1.33, label, fontsize=8, ha="center",
        va="center", color=P[ck][2], fontweight="bold", zorder=5)
    if i < len(ci)-1:
        arrow_h(bx+1.52, BY3+1.33, bx+1.78, ck)

# Monitoring right half
box(11.9, BY3, 11.7, BH3, "pink", "MONITORING  —  Prometheus + Grafana  (docker-compose)",
    ["API exposes /metrics endpoint  ·  scraped every 15s  ·  13-panel Grafana dashboard"])

mon = [("Prometheus\n:9090","scrapes\n/metrics"),
       ("Grafana\n:3000","13 panels\ndashboard"),
       ("Predictions\nCounter","high/low\nrisk split"),
       ("Latency\nHistogram","p50·p95·p99\nper call"),
       ("Structured\nLogs","age·cp·prob\nper request")]
for i, (top, bot) in enumerate(mon):
    bx = 12.15 + i * 2.28
    inner = FancyBboxPatch((bx, BY3+0.38), 2.0, 1.9,
        boxstyle="round,pad=0,rounding_size=0.1",
        linewidth=1.5, edgecolor=P["pink"][1], facecolor="#FDF6FB", zorder=4)
    ax.add_patch(inner)
    ax.text(bx+1.0, BY3+1.52, top, fontsize=8, ha="center",
        va="center", color=P["pink"][2], fontweight="bold", zorder=5)
    ax.text(bx+1.0, BY3+0.76, bot, fontsize=7, ha="center",
        va="center", color="#9D174D", zorder=5)

# Arrow FastAPI → Monitoring (go right then down)
arrow_angled(pos2[0]+BW2/2, BY2, 18.0, BY3+BH3, "pink", "/metrics")

# Arrow GitHub Actions → row1 (go right then up)
arrow_angled(5.8, BY3+BH3, positions[4]+BW/2, BY, "orange", "smoke test")

# ═══════════════════════════════════════════════════════════════════════════
# BOTTOM BAR
# ═══════════════════════════════════════════════════════════════════════════
ax.plot([0.5, 27.5], [2.95, 2.95], color="#E5E7EB", lw=1.5)

# stat chips
chips = [
    ("297 patients", "blue"), ("18 features", "purple"),
    ("3 models", "green"), ("ROC-AUC 0.9464", "green"),
    ("25 tests", "red"), ("11 CI steps", "orange"),
    ("2 K8s replicas", "indigo"), ("13 Grafana panels", "pink"),
]
cx = 0.4
for txt, ck in chips:
    bg, border, _ = P[ck]
    w = len(txt)*0.1 + 0.3
    r = FancyBboxPatch((cx, 2.5), w, 0.38,
        boxstyle="round,pad=0,rounding_size=0.09",
        linewidth=1.5, edgecolor=border, facecolor=bg, zorder=3)
    ax.add_patch(r)
    ax.text(cx+w/2, 2.69, txt, fontsize=8, ha="center", va="center",
        color=border, fontweight="bold", zorder=4)
    cx += w + 0.2

# Live URL bar
url_rect = FancyBboxPatch((3.5, 1.8), 21.0, 0.52,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=2, edgecolor=P["emerald"][1], facecolor=P["emerald"][0], zorder=3)
ax.add_patch(url_rect)
ax.text(14, 2.06,
    "Live API:   https://heart-disease-mlops-rcg4.onrender.com/docs",
    fontsize=10.5, ha="center", va="center",
    color=P["emerald"][2], fontweight="bold", zorder=4)

# outer border
outer = FancyBboxPatch((0.15, 1.65), 27.7, 11.3,
    boxstyle="round,pad=0,rounding_size=0.25",
    linewidth=2, edgecolor="#D1D5DB", facecolor="none", zorder=1)
ax.add_patch(outer)

# GitHub repo label
ax.text(26.5, 2.69, "github.com/Swathi-A01/\nheart-disease-mlops",
    fontsize=8, ha="center", va="center", color=P["blue"][1],
    fontweight="bold")

plt.tight_layout(pad=0.1)
plt.savefig(OUT, dpi=180, bbox_inches="tight",
    facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size//1024}KB)")
