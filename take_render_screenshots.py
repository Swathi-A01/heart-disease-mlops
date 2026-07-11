"""
Retake Render screenshots with proper wait times.
"""
import time
import json
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

SHOTS = Path("screenshots")
SHOTS.mkdir(exist_ok=True)

RENDER_URL = "https://heart-disease-mlops-rcg4.onrender.com"

SAMPLE_HIGH = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
    "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
    "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7
}
SAMPLE_LOW = {
    "age": 35, "sex": 0, "cp": 1, "trestbps": 105, "chol": 180,
    "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
    "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_context(viewport={"width": 1400, "height": 900}).new_page()
    page.set_default_timeout(60000)

    # ── Swagger UI ────────────────────────────────────────────────────────────
    print("Swagger UI...")
    page.goto(f"{RENDER_URL}/docs", wait_until="networkidle", timeout=60000)
    time.sleep(4)
    page.screenshot(path=str(SHOTS / "render_swagger_ui.png"), full_page=False)
    print("  saved render_swagger_ui.png")

    # Expand /predict endpoint in Swagger
    try:
        page.click("text=/predict", timeout=5000)
        time.sleep(1.5)
        page.screenshot(path=str(SHOTS / "render_swagger_predict.png"), full_page=False)
        print("  saved render_swagger_predict.png")
    except Exception:
        pass

    # ── /health as styled JSON page ───────────────────────────────────────────
    print("Health + model-info + stats...")
    for endpoint, fname in [
        ("/health",     "render_health.png"),
        ("/model-info", "render_model_info.png"),
        ("/stats",      "render_stats.png"),
    ]:
        try:
            req = urllib.request.Request(f"{RENDER_URL}{endpoint}")
            resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            resp_pretty = json.dumps(resp, indent=2)
            html = f"""<html><body style="background:#1e1e2e;color:#cdd6f4;
                font-family:monospace;font-size:15px;padding:30px;margin:0;">
                <div style="background:#313244;border-radius:10px;padding:25px;max-width:800px;">
                <div style="color:#89b4fa;font-size:16px;font-weight:bold;margin-bottom:12px;">
                GET {RENDER_URL}{endpoint}</div>
                <div style="color:#6c7086;font-size:12px;margin-bottom:16px;">
                Live deployment on Render — https://render.com</div>
                <pre style="margin:0;color:#a6e3a1;font-size:14px;line-height:1.6;">{resp_pretty}</pre>
                </div></body></html>"""
            page.set_content(html)
            time.sleep(1)
            page.screenshot(path=str(SHOTS / fname), full_page=False)
            print(f"  saved {fname}")
        except Exception as e:
            print(f"  {endpoint} failed: {e}")

    # ── Predict responses ─────────────────────────────────────────────────────
    print("Predict responses...")
    for label, sample, fname in [
        ("High Risk Patient (67M, multiple red flags)", SAMPLE_HIGH, "render_predict_high.png"),
        ("Low Risk Patient (35F, good cardiac metrics)", SAMPLE_LOW,  "render_predict_low.png"),
    ]:
        try:
            req = urllib.request.Request(
                f"{RENDER_URL}/predict",
                data=json.dumps(sample).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            resp_pretty = json.dumps(resp, indent=2)
            req_pretty  = json.dumps(sample, indent=2)

            risk_color = "#f38ba8" if resp.get("risk") == "high" else "#a6e3a1"

            html = f"""<html><body style="background:#1e1e2e;color:#cdd6f4;
                font-family:monospace;font-size:14px;padding:30px;margin:0;">
                <div style="background:#313244;border-radius:10px;padding:25px;max-width:900px;">
                <div style="color:#89b4fa;font-size:16px;font-weight:bold;margin-bottom:4px;">
                POST {RENDER_URL}/predict</div>
                <div style="color:#6c7086;font-size:11px;margin-bottom:16px;">
                {label} — Live Render Deployment</div>
                <div style="display:flex;gap:20px;">
                <div style="flex:1;">
                <div style="color:#cba6f7;font-weight:bold;margin-bottom:8px;">Request Body:</div>
                <pre style="margin:0;background:#1e1e2e;padding:12px;border-radius:6px;
                     color:#cdd6f4;font-size:12px;line-height:1.5;">{req_pretty}</pre>
                </div>
                <div style="flex:1;">
                <div style="color:#cba6f7;font-weight:bold;margin-bottom:8px;">Response:</div>
                <pre style="margin:0;background:#1e1e2e;padding:12px;border-radius:6px;
                     color:{risk_color};font-size:13px;line-height:1.7;">{resp_pretty}</pre>
                </div>
                </div>
                </div></body></html>"""
            page.set_content(html)
            time.sleep(1)
            page.screenshot(path=str(SHOTS / fname), full_page=False)
            print(f"  saved {fname}")
        except Exception as e:
            print(f"  predict failed: {e}")

    # ── Render dashboard URL page ─────────────────────────────────────────────
    print("Render deployment URL card...")
    html_url = f"""<html><body style="background:#1e1e2e;color:#cdd6f4;
        font-family:monospace;font-size:15px;padding:40px;margin:0;">
        <div style="background:#313244;border-radius:12px;padding:30px;max-width:750px;
             border:1px solid #45475a;">
        <div style="color:#a6e3a1;font-size:22px;font-weight:bold;margin-bottom:10px;">
        Heart Disease Risk API — Live on Render</div>
        <div style="color:#6c7086;font-size:13px;margin-bottom:20px;">
        Cloud deployment · Auto-deployed from GitHub · HTTPS enabled</div>
        <div style="margin-bottom:12px;">
        <span style="color:#89b4fa;font-weight:bold;">Swagger UI: </span>
        <span style="color:#cdd6f4;">{RENDER_URL}/docs</span></div>
        <div style="margin-bottom:12px;">
        <span style="color:#89b4fa;font-weight:bold;">/health:    </span>
        <span style="color:#a6e3a1;">{{"status":"ok","model":"heart-disease-classifier","version":"1.0.0"}}</span></div>
        <div style="margin-bottom:20px;">
        <span style="color:#89b4fa;font-weight:bold;">/predict:   </span>
        <span style="color:#a6e3a1;">{{"prediction":1,"probability":0.9982,"risk":"high",...}}</span></div>
        <div style="background:#1e1e2e;padding:15px;border-radius:8px;font-size:12px;color:#6c7086;">
        Platform: Render · Region: Singapore · Instance: Free tier<br/>
        Auto-deploys on every push to main branch · TLS certificate included</div>
        </div></body></html>"""
    page.set_content(html_url)
    time.sleep(1)
    page.screenshot(path=str(SHOTS / "render_deployment_card.png"), full_page=False)
    print("  saved render_deployment_card.png")

    browser.close()

print(f"\nAll Render screenshots saved to {SHOTS.resolve()}")
for f in sorted(SHOTS.glob("render_*.png")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name} — {size_kb}KB")
