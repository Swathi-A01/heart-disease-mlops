"""
Generate a clean MLOps-themed cover illustration.
Saves to plots/mlops_cover.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

OUT = Path("plots/mlops_cover.png")

BG     = "#FAFBFC"
NAVY   = "#1A1A2E"
BLUE   = "#58A6FF"
GREEN  = "#3FB950"
PURPLE = "#BC8CFF"
CORAL  = "#F78166"
TEAL   = "#56D364"
AMBER  = "#D29922"
MUTED  = "#8B949E"

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# ── background subtle grid dots ──
for x in np.arange(0.5, 14, 0.9):
    for y in np.arange(0.5, 7, 0.9):
        ax.plot(x, y, '.', color="#E1E4E8", markersize=2, zorder=1)

# ── pipeline nodes ────────────────────────────────────────────────────────────
nodes = [
    (1.4,  3.5, "DATA",        "heart.csv\n303 patients",         BLUE,   "📊"),
    (3.4,  3.5, "FEATURES",    "StandardScaler\nOHE + 5 derived", PURPLE, "⚙"),
    (5.4,  3.5, "TRAINING",    "LR · RF · XGBoost\nMLflow logs",  GREEN,  "🧠"),
    (7.4,  3.5, "API",         "FastAPI\n/predict · /batch",      CORAL,  "⚡"),
    (9.4,  3.5, "DOCKER",      "python:3.11-slim\nself-contained", TEAL,  "🐳"),
    (11.4, 3.5, "KUBERNETES",  "2 replicas\nLoadBalancer",        BLUE,   "☸"),
    (13.0, 5.5, "MONITORING",  "Prometheus\nGrafana",             AMBER,  "📈"),
    (13.0, 1.5, "CI/CD",       "GitHub Actions\n25 tests",        CORAL,  "🔄"),
]

BOX_W = 1.6
BOX_H = 1.5

def draw_node(x, y, title, desc, color, icon):
    # shadow
    shadow = FancyBboxPatch((x - BOX_W/2 + 0.05, y - BOX_H/2 - 0.05),
        BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.15",
        linewidth=0, facecolor="#D0D7DE", alpha=0.5, zorder=2)
    ax.add_patch(shadow)
    # card
    card = FancyBboxPatch((x - BOX_W/2, y - BOX_H/2),
        BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.15",
        linewidth=1.5, edgecolor=color, facecolor="white", alpha=1.0, zorder=3)
    ax.add_patch(card)
    # top color strip
    strip = FancyBboxPatch((x - BOX_W/2, y + BOX_H/2 - 0.32),
        BOX_W, 0.32, boxstyle="round,pad=0,rounding_size=0.1",
        linewidth=0, facecolor=color, alpha=0.9, zorder=4)
    ax.add_patch(strip)
    # title
    ax.text(x, y + BOX_H/2 - 0.16, title, fontsize=8, fontweight="bold",
            ha="center", va="center", color="white", zorder=5)
    # description
    ax.text(x, y - 0.12, desc, fontsize=6.5, ha="center", va="center",
            color="#444", zorder=5, linespacing=1.4)

# draw all nodes
for nx, ny, title, desc, color, icon in nodes:
    draw_node(nx, ny, title, desc, color, icon)

# ── arrows between main pipeline nodes ───────────────────────────────────────
main_nodes = [(n[0], n[1]) for n in nodes[:6]]
for i in range(len(main_nodes)-1):
    x1, y1 = main_nodes[i]
    x2, y2 = main_nodes[i+1]
    ax.annotate("", xy=(x2 - BOX_W/2, y2),
                xytext=(x1 + BOX_W/2, y1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                lw=1.5, mutation_scale=12), zorder=2)

# Arrow from API to monitoring and CI/CD
ax.annotate("", xy=(13.0 - BOX_W/2, 5.5),
            xytext=(7.4 + BOX_W/2, 3.8),
            arrowprops=dict(arrowstyle="-|>", color=AMBER,
                            lw=1.2, mutation_scale=10,
                            connectionstyle="arc3,rad=-0.3"), zorder=2)
ax.annotate("", xy=(13.0 - BOX_W/2, 1.5),
            xytext=(5.4 + BOX_W/2, 3.2),
            arrowprops=dict(arrowstyle="-|>", color=CORAL,
                            lw=1.2, mutation_scale=10,
                            connectionstyle="arc3,rad=0.3"), zorder=2)

# ── MLflow badge (below training) ────────────────────────────────────────────
mlf = FancyBboxPatch((4.55, 1.6), 1.7, 0.55,
    boxstyle="round,pad=0,rounding_size=0.1",
    linewidth=1, edgecolor="#0194E2", facecolor="#E8F4FD", zorder=3)
ax.add_patch(mlf)
ax.text(5.4, 1.875, "MLflow Experiment Tracking",
        fontsize=7, ha="center", va="center", color="#0194E2",
        fontweight="bold", zorder=5)
ax.annotate("", xy=(5.4, 2.75), xytext=(5.4, 2.15),
            arrowprops=dict(arrowstyle="-|>", color="#0194E2",
                            lw=1.2, mutation_scale=10), zorder=2)

# ── title text ────────────────────────────────────────────────────────────────
ax.text(7.0, 6.55, "Heart Disease Risk Prediction  —  MLOps Pipeline",
        fontsize=15, fontweight="bold", ha="center", va="center",
        color=NAVY, zorder=5)
ax.text(7.0, 6.12, "Data  →  Feature Engineering  →  Training  →  API  →  Docker  →  Kubernetes  →  Monitoring",
        fontsize=8.5, ha="center", va="center", color=MUTED,
        style="italic", zorder=5)

# ── accuracy badge ────────────────────────────────────────────────────────────
badge = FancyBboxPatch((0.15, 0.2), 2.5, 0.9,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=1.2, edgecolor=GREEN, facecolor="#EAFBEA", zorder=3)
ax.add_patch(badge)
ax.text(1.4, 0.80, "Best Model: Random Forest",
        fontsize=7.5, fontweight="bold", ha="center", va="center",
        color="#1A7F37", zorder=5)
ax.text(1.4, 0.38, "Test ROC-AUC: 0.9464  ·  Accuracy: 83.3%",
        fontsize=7, ha="center", va="center", color="#2D8A4E", zorder=5)

# ── dataset badge ─────────────────────────────────────────────────────────────
db = FancyBboxPatch((5.5, 0.2), 3.0, 0.9,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=1.2, edgecolor=BLUE, facecolor="#EBF5FF", zorder=3)
ax.add_patch(db)
ax.text(7.0, 0.80, "UCI Heart Disease (Cleveland)",
        fontsize=7.5, fontweight="bold", ha="center", va="center",
        color="#1A5FA8", zorder=5)
ax.text(7.0, 0.38, "297 patients  ·  13 features  ·  Binary target",
        fontsize=7, ha="center", va="center", color="#2A71B8", zorder=5)

# ── test badge ────────────────────────────────────────────────────────────────
tb = FancyBboxPatch((11.35, 0.2), 2.5, 0.9,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=1.2, edgecolor=CORAL, facecolor="#FFF0EE", zorder=3)
ax.add_patch(tb)
ax.text(12.6, 0.80, "CI/CD: 25 Tests Passing",
        fontsize=7.5, fontweight="bold", ha="center", va="center",
        color="#B04040", zorder=5)
ax.text(12.6, 0.38, "flake8  ·  pytest  ·  GitHub Actions",
        fontsize=7, ha="center", va="center", color="#B04040", zorder=5)

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
