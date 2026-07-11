"""
Tools-used diagram.
Row 1: graphviz (real logos, linear chain — works perfectly)
Row 2: matplotlib with colored icon-style boxes
"""
import os
os.environ["PATH"] += ":/opt/homebrew/bin"

from diagrams import Diagram, Edge
from diagrams.programming.language import Python
from diagrams.onprem.database import Mysql
from diagrams.generic.storage import Storage
from diagrams.programming.framework import FastAPI

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import io
from pathlib import Path
import numpy as np

BASE  = Path("screenshots")
FINAL = BASE / "tools_used.png"

# ── ROW 1: graphviz ───────────────────────────────────────────────────────
GA = {"bgcolor":"white","pad":"0.6","splines":"ortho",
      "nodesep":"0.7","ranksep":"1.4","fontname":"Helvetica Neue","rankdir":"LR"}
NA = {"fontsize":"13","fontname":"Helvetica Neue","fixedsize":"true","width":"1.7","height":"1.7"}
EA = {"fontsize":"12","fontname":"Helvetica Neue","color":"#444444","penwidth":"2.0"}

with Diagram("", filename=str(BASE/"row1"), outformat="png",
             show=False, direction="LR",
             graph_attr={**GA,"fontsize":"1"}, node_attr=NA, edge_attr=EA):
    uci     = Storage("UCI Dataset\n303 patients")
    py      = Python("Python 3.11\npandas · numpy")
    sklearn = Python("scikit-learn\nColumnTransformer")
    models  = Python("LR · RF\nXGBoost")
    mlflow  = Mysql("MLflow\n+ SQLite")
    pkl     = Storage("pipeline.pkl\njoblib")
    api     = FastAPI("FastAPI\n+ Uvicorn")

    uci     >> Edge(label="download") >> py
    py      >> Edge(label="heart.csv") >> sklearn
    sklearn >> Edge(label="X_train") >> models
    models  >> Edge(label="log runs", style="dashed") >> mlflow
    models  >> Edge(label="best model", color="#16a34a", style="bold") >> pkl
    pkl     >> Edge(label="load") >> api

# ── ROW 2: matplotlib deployment section ──────────────────────────────────
TOOLS = [
    # Row 1: FastAPI → Docker → K8s  (y=3.8)
    (1.5,  3.8, "FastAPI\n+ Uvicorn",    "F",  "#ECFDF5", "#059669"),
    (5.5,  3.8, "Docker\n3.11-slim",     "D",  "#E0F2FE", "#0284C7"),
    (9.5,  4.8, "Kubernetes\n2 replicas","K",  "#EEF2FF", "#4F46E5"),
    (9.5,  2.8, "Render\nCloud HTTPS",   "R",  "#F0FDF4", "#16A34A"),
    # Row 2: Prometheus → Grafana (y=3.8, after Docker)
    (13.5, 3.8, "Prometheus\n:9090",     "P",  "#FEF2F2", "#DC2626"),
    (17.0, 3.8, "Grafana\n13 panels",    "G",  "#FFF7ED", "#EA580C"),
    # CI/CD row (y=1.2)
    (1.5,  1.2, "GitHub Actions\nCI/CD", ">",  "#EFF6FF", "#2563EB"),
    (5.5,  1.2, "Pytest\n25 tests",      "T",  "#FEF9C3", "#CA8A04"),
]

ARROWS = [
    (1.5+1.1, 3.5,  5.0-1.1, 3.5,  "build",          "#0284C7"),
    (5.0+1.1, 4.0,  8.5-1.1, 4.5,  "local K8s",      "#4F46E5"),
    (5.0+1.1, 3.0,  8.5-1.1, 2.5,  "cloud deploy",   "#16A34A"),
    (1.5+1.1, 3.0,  12.5-1.1,4.5,  "/metrics",       "#DC2626"),
    (12.5+1.1,4.5,  16.5,    4.5,   "",               "#EA580C"),   # prom → grafana
    (12.5+1.1,2.5,  16.5,    2.5,   "",               "#EA580C"),
    (1.5+1.1, 1.0,  5.0-1.1, 1.0,  "lint→test→train","#2563EB"),
]

# merge prom and grafana arrows
ARROWS = [
    # FastAPI → Docker
    (2.6,  3.8,  4.4,  3.8,  "build",           "#0284C7"),
    # Docker → K8s (up)
    (6.6,  4.2,  8.4,  4.8,  "local K8s",       "#4F46E5"),
    # Docker → Render (down)
    (6.6,  3.4,  8.4,  2.8,  "cloud deploy",    "#16A34A"),
    # FastAPI → Prometheus
    (2.6,  3.5,  12.4, 3.8,  "/metrics",        "#DC2626"),
    # Prometheus → Grafana
    (14.6, 3.8,  15.9, 3.8,  "visualise",       "#EA580C"),
    # GitHub → Pytest
    (2.6,  1.2,  4.4,  1.2,  "lint→test→train", "#2563EB"),
]

FIG_W, FIG_H = 20, 6.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Grafana already in TOOLS above

# Draw nodes
for (cx, cy, lbl, sym, fill, border) in TOOLS:
    # outer card
    card = FancyBboxPatch((cx-1.1, cy-0.85), 2.2, 1.7,
        boxstyle="round,pad=0,rounding_size=0.18",
        linewidth=2, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(card)
    # color top bar
    topbar = FancyBboxPatch((cx-1.1, cy+0.55), 2.2, 0.3,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0, facecolor=border, alpha=0.85, zorder=4)
    ax.add_patch(topbar)
    # symbol in top bar
    ax.text(cx, cy+0.7, sym, fontsize=13, ha="center", va="center",
        color="white", fontweight="bold", zorder=5)
    # label
    ax.text(cx, cy-0.08, lbl, fontsize=8.5, ha="center", va="center",
        color="#1E293B", fontweight="bold", zorder=5, linespacing=1.4)

# Draw arrows
for (x1, y1, x2, y2, lbl, col) in ARROWS:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=2,
        mutation_scale=14, connectionstyle="arc3,rad=0.0"), zorder=4)
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.2
        ax.text(mx, my, lbl, fontsize=7.5, color=col, ha="center",
            style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor=col, linewidth=0.8, alpha=0.95), zorder=6)

plt.tight_layout(pad=0)
row2_buf = io.BytesIO()
plt.savefig(row2_buf, dpi=160, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
row2_buf.seek(0)
img2 = Image.open(row2_buf).copy()

# ── Combine ────────────────────────────────────────────────────────────────
img1 = Image.open(BASE/"row1.png")
W = max(img1.width, img2.width)

def resize_w(img):
    return img.resize((W, int(img.height * W / img.width)), Image.LANCZOS)

img1 = resize_w(img1)
img2 = resize_w(img2)

def section_bar(img, label):
    H = 46
    new = Image.new("RGB", (img.width, img.height + H), "white")
    d = ImageDraw.Draw(new)
    d.rectangle([0, 0, img.width, H], fill=(241, 245, 249))
    d.line([0, H, img.width, H], fill=(203, 213, 225), width=2)
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 21)
    except Exception:
        fnt = ImageFont.load_default()
    d.text((16, 11), label, fill=(30, 41, 59), font=fnt)
    new.paste(img, (0, H))
    return new

img1 = section_bar(img1, "● Main Pipeline:  Data → Feature Engineering → Training (LR · RF · XGBoost) → MLflow → FastAPI")
img2 = section_bar(img2, "● Deployment + CI/CD + Monitoring:  Docker → Kubernetes / Render  ·  Prometheus → Grafana  ·  GitHub Actions")

PAD = 24
TH  = 72
canvas_h = TH + img1.height + PAD + img2.height + PAD
canvas = Image.new("RGB", (W + 2*PAD, canvas_h), "white")
d = ImageDraw.Draw(canvas)

try:
    tf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    sf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except Exception:
    tf = sf = ImageFont.load_default()

d.text((W//2+PAD, 12), "Tools Used — Heart Disease MLOps Pipeline",
       fill=(15,23,42), font=tf, anchor="mt")
d.text((W//2+PAD, 50),
    "Python · scikit-learn · XGBoost · MLflow · FastAPI · Docker · Kubernetes · Render · Prometheus · Grafana · GitHub Actions · Pytest",
    fill=(100,116,139), font=sf, anchor="mt")
d.line([PAD, TH-2, W+PAD, TH-2], fill=(226,232,240), width=2)
canvas.paste(img1, (PAD, TH))
canvas.paste(img2, (PAD, TH + img1.height + PAD))
d.rectangle([2,2,canvas.width-3,canvas.height-3], outline=(203,213,225), width=3)
canvas.save(FINAL, dpi=(160,160))
(BASE/"row1.png").unlink(missing_ok=True)
print(f"Saved: {FINAL}  ({FINAL.stat().st_size//1024}KB)")
