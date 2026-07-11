"""
Tools diagram — 4 separate graphviz chains, all horizontal, combined with PIL.
Each chain is purely linear so graphviz can't mess up the layout.
"""
import os
os.environ["PATH"] += ":/opt/homebrew/bin"

from diagrams import Diagram, Edge
from diagrams.programming.language import Python
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.container import Docker
from diagrams.onprem.network import Nginx
from diagrams.onprem.ci import GithubActions
from diagrams.generic.storage import Storage
from diagrams.programming.framework import FastAPI
from diagrams.k8s.compute import Deployment
from diagrams.k8s.network import Service
from diagrams.onprem.mlops import Mlflow
from diagrams.generic.database import SQL

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BASE  = Path("screenshots")
FINAL = BASE / "tools_used.png"

GA  = {"bgcolor":"white","pad":"0.5","splines":"ortho","nodesep":"0.8","ranksep":"1.3","fontname":"Helvetica Neue","rankdir":"LR"}
GAS = {"bgcolor":"white","pad":"0.3","splines":"ortho","nodesep":"0.6","ranksep":"1.0","fontname":"Helvetica Neue","rankdir":"LR"}
NA  = {"fontsize":"13","fontname":"Helvetica Neue","fixedsize":"true","width":"1.6","height":"1.6"}
NAS = {"fontsize":"13","fontname":"Helvetica Neue","fixedsize":"true","width":"1.2","height":"1.2"}
EA = {"fontsize":"12","fontname":"Helvetica Neue",
      "color":"#444444","penwidth":"2.0"}

# ── Chain 1: Main pipeline ─────────────────────────────────────────────────
with Diagram("", filename=str(BASE/"c1"), outformat="png", show=False,
             direction="LR", graph_attr={**GA,"fontsize":"1"},
             node_attr=NA, edge_attr=EA):
    Storage("UCI Dataset\n303 patients") \
    >> Edge(label="download") \
    >> Python("Python 3.11\npandas · numpy") \
    >> Edge(label="heart.csv") \
    >> Python("scikit-learn\nColumnTransformer") \
    >> Edge(label="X_train") \
    >> Python("LR · RF · XGBoost") \
    >> Edge(label="best model", color="#16a34a", style="bold") \
    >> Storage("pipeline.pkl\njoblib") \
    >> Edge(label="load") \
    >> FastAPI("FastAPI\n+ Uvicorn")

# ── Chain 2: Experiment tracking (separate so it doesn't pull layout) ──────
with Diagram("", filename=str(BASE/"c2"), outformat="png", show=False,
             direction="LR", graph_attr={**GA,"fontsize":"1"},
             node_attr=NA, edge_attr=EA):
    Python("LR · RF · XGBoost") \
    >> Edge(label="log params + metrics + artifacts", style="dashed") \
    >> Mlflow("MLflow\nExperiment Tracking") \
    >> Edge(label="stored in", style="dashed") \
    >> SQL("SQLite\nmlflow.db")

# ── Chain 3: Docker → Kubernetes + Render ─────────────────────────────────
with Diagram("", filename=str(BASE/"c3"), outformat="png", show=False,
             direction="LR", graph_attr={**GA,"fontsize":"1"},
             node_attr=NA, edge_attr=EA):
    FastAPI("FastAPI") \
    >> Edge(label="build") \
    >> Docker("Docker\n3.11-slim") \
    >> Edge(label="local K8s") \
    >> Deployment("Kubernetes\n2 replicas") \
    >> Edge(label="LoadBalancer") \
    >> Service("port 80→8000")

with Diagram("", filename=str(BASE/"c3b"), outformat="png", show=False,
             direction="LR", graph_attr={**GA,"fontsize":"1"},
             node_attr=NA, edge_attr=EA):
    FastAPI("FastAPI\n+ Uvicorn") \
    >> Edge(label="Dockerfile.render") \
    >> Docker("Docker\n3.11-slim") \
    >> Edge(label="auto-deploy on push") \
    >> Nginx("Render\nCloud HTTPS\n(free tier)")

# ── Chain 4: Monitoring ────────────────────────────────────────────────────
with Diagram("", filename=str(BASE/"c4"), outformat="png", show=False,
             direction="LR", graph_attr={**GA,"fontsize":"1"},
             node_attr=NA, edge_attr=EA):
    FastAPI("FastAPI") \
    >> Edge(label="exposes /metrics endpoint") \
    >> Prometheus("Prometheus\n:9090") \
    >> Edge(label="scrapes every 15s") \
    >> Grafana("Grafana\n13-panel dashboard")

# ── Chain 5: CI/CD ─────────────────────────────────────────────────────────
with Diagram("", filename=str(BASE/"c5"), outformat="png", show=False,
             direction="LR", graph_attr={**GA,"fontsize":"1"},
             node_attr=NA, edge_attr=EA):
    GithubActions("GitHub Actions\npush → ubuntu-latest VM") \
    >> Edge(label="flake8 → pytest 25 tests → train → upload artifacts") \
    >> Python("Pytest\n25 tests pass")

# ── Combine all chains with PIL ────────────────────────────────────────────
chains = ["c1","c2","c3","c3b","c4","c5"]
imgs   = [Image.open(BASE/f"{c}.png") for c in chains]

labels = [
    "① Data Acquisition  →  Feature Engineering  →  Model Training  →  Best Model  →  FastAPI",
    "② Experiment Tracking  (all runs logged with params, metrics, plots)",
    "③ Containerisation  →  Local Kubernetes Deployment  (2 replicas, LoadBalancer)",
    "④ Cloud Deployment:  FastAPI  →  Docker (Dockerfile.render)  →  Render  (auto-deploy on git push)",
    "⑤ Monitoring  (Prometheus scrapes /metrics, Grafana 13-panel dashboard)",
    "⑥ CI/CD  (GitHub Actions: lint → test → train → upload artifacts on every push)",
]

# Auto-crop white margins from each image
def crop_white(img, margin=20):
    import numpy as np
    arr = np.array(img.convert("RGB"))
    mask = ~((arr[:,:,0]>245) & (arr[:,:,1]>245) & (arr[:,:,2]>245))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin,rmax = np.where(rows)[0][[0,-1]]
        cmin,cmax = np.where(cols)[0][[0,-1]]
        rmin = max(0, rmin-margin)
        rmax = min(img.height, rmax+margin)
        cmin = max(0, cmin-margin)
        cmax = min(img.width, cmax+margin)
        return img.crop((cmin, rmin, cmax, rmax))
    return img

imgs = [crop_white(img) for img in imgs]

# Cap height — 2-node chains render too large; clamp to 350px tall
MAX_H = 350
def cap_height(img):
    if img.height > MAX_H:
        return img.resize((int(img.width * MAX_H / img.height), MAX_H), Image.LANCZOS)
    return img
imgs = [cap_height(img) for img in imgs]
W = max(img.width for img in imgs)

def resize_w(img):
    return img.resize((W, int(img.height * W / img.width)), Image.LANCZOS)

BAR_H = 46
PAD   = 20

def add_label(img, text):
    img = resize_w(img)
    new = Image.new("RGB", (img.width, img.height + BAR_H), "white")
    d   = ImageDraw.Draw(new)
    d.rectangle([0, 0, img.width, BAR_H], fill=(241, 245, 249))
    d.line([0, BAR_H, img.width, BAR_H], fill=(203, 213, 225), width=2)
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    except Exception:
        fnt = ImageFont.load_default()
    d.text((16, 13), text, fill=(30, 41, 59), font=fnt)
    new.paste(img, (0, BAR_H))
    return new

labelled = [add_label(img, lbl) for img, lbl in zip(imgs, labels)]

TH = 76
total_h = TH + sum(img.height + PAD for img in labelled)
canvas  = Image.new("RGB", (W + 2*PAD, total_h), "white")
d       = ImageDraw.Draw(canvas)

try:
    tf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    sf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except Exception:
    tf = sf = ImageFont.load_default()

cx = W // 2 + PAD
d.text((cx, 12), "Tools Used — Heart Disease MLOps Pipeline",
       fill=(15, 23, 42), font=tf, anchor="mt")
d.text((cx, 52),
    "Python · scikit-learn · XGBoost · MLflow · FastAPI · Docker · Kubernetes · Render · Prometheus · Grafana · GitHub Actions",
    fill=(100, 116, 139), font=sf, anchor="mt")
d.line([PAD, TH-2, W+PAD, TH-2], fill=(226, 232, 240), width=2)

y = TH
for img in labelled:
    canvas.paste(img, (PAD, y))
    y += img.height + PAD
    # light separator
    d.line([PAD, y-PAD//2, W+PAD, y-PAD//2], fill=(241, 245, 249), width=1)

d.rectangle([2, 2, canvas.width-3, canvas.height-3],
            outline=(203, 213, 225), width=3)
canvas.save(FINAL, dpi=(160, 160))

for c in chains:
    (BASE/f"{c}.png").unlink(missing_ok=True)

print(f"Saved: {FINAL}  ({FINAL.stat().st_size//1024}KB)")
