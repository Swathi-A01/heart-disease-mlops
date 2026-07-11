"""
Take screenshots of the live Render deployment.
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


def take_render_screenshots(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1400, "height": 900}).new_page()
        page.set_default_timeout(60000)

        # Swagger UI
        print(f"Opening {url}/docs ...")
        page.goto(f"{url}/docs", wait_until="domcontentloaded")
        time.sleep(5)
        page.screenshot(path=str(SHOTS / "render_swagger_ui.png"))
        print("  saved: render_swagger_ui.png")

        # /health in browser
        page.goto(f"{url}/health", wait_until="domcontentloaded")
        time.sleep(2)
        page.screenshot(path=str(SHOTS / "render_health.png"))
        print("  saved: render_health.png")

        # /model-info
        page.goto(f"{url}/model-info", wait_until="domcontentloaded")
        time.sleep(2)
        page.screenshot(path=str(SHOTS / "render_model_info.png"))
        print("  saved: render_model_info.png")

        # Styled predict responses
        for label, sample, fname in [
            ("High Risk Patient", SAMPLE_HIGH, "render_predict_high.png"),
            ("Low Risk Patient",  SAMPLE_LOW,  "render_predict_low.png"),
        ]:
            try:
                req = urllib.request.Request(
                    f"{url}/predict",
                    data=json.dumps(sample).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
                resp_pretty = json.dumps(resp, indent=2)
                req_pretty  = json.dumps(sample, indent=2)

                html = f"""<html><body style="background:#1e1e2e;color:#cdd6f4;
                    font-family:monospace;font-size:15px;padding:30px;">
                    <h2 style="color:#89b4fa;">POST {url}/predict</h2>
                    <h3 style="color:#a6e3a1;">{label} — Request:</h3>
                    <pre style="background:#313244;padding:15px;border-radius:8px">{req_pretty}</pre>
                    <h3 style="color:#a6e3a1;">Response:</h3>
                    <pre style="background:#313244;padding:15px;border-radius:8px;color:#a6e3a1">{resp_pretty}</pre>
                    </body></html>"""
                page.set_content(html)
                time.sleep(1)
                page.screenshot(path=str(SHOTS / fname))
                print(f"  saved: {fname}")
            except Exception as e:
                print(f"  prediction failed: {e}")

        browser.close()

    print(f"\nRender screenshots saved to {SHOTS.resolve()}")


if __name__ == "__main__":
    take_render_screenshots(RENDER_URL)
