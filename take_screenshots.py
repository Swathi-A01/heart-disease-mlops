"""
Continue taking screenshots from MLflow onwards (API shots already done).
"""
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright

SHOTS = Path("screenshots")
SHOTS.mkdir(exist_ok=True)


def shot(page, name, wait=2):
    time.sleep(wait)
    path = str(SHOTS / f"{name}.png")
    page.screenshot(path=path, full_page=False)
    print(f"  saved: {path}")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.set_default_timeout(20000)

    # ── 1. MLflow ─────────────────────────────────────────────────────────
    print("MLflow...")
    page.goto("http://localhost:5000", wait_until="domcontentloaded")
    time.sleep(5)
    shot(page, "07_mlflow_home")

    page.goto("http://localhost:5000/#/experiments/1", wait_until="domcontentloaded")
    time.sleep(5)
    shot(page, "08_mlflow_runs_list")

    # Try clicking a run
    try:
        links = page.query_selector_all("a[href*='/runs/']")
        if links:
            links[0].click()
            time.sleep(3)
            shot(page, "10_mlflow_run_detail")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)
            shot(page, "11_mlflow_run_artifacts")
    except Exception as e:
        print(f"  run detail skipped: {e}")

    # ── 2. GitHub Actions ─────────────────────────────────────────────────
    print("GitHub Actions...")
    page.goto(
        "https://github.com/Swathi-A01/heart-disease-mlops/actions",
        wait_until="domcontentloaded",
        timeout=30000
    )
    time.sleep(4)
    shot(page, "12_github_actions_list")

    # Latest successful run detail
    try:
        # Find first success run link
        run_link = page.query_selector(".workflow-list-item-link")
        if not run_link:
            run_link = page.query_selector("a[href*='/actions/runs/']")
        if run_link:
            run_link.click()
            time.sleep(3)
            shot(page, "13_github_actions_run_detail")
            # Try expanding job
            try:
                job = page.query_selector(".js-job-name")
                if job:
                    job.click()
                    time.sleep(2)
                    shot(page, "14_github_actions_job_steps")
            except Exception:
                pass
    except Exception as e:
        print(f"  run detail: {e}")

    # GitHub repo main page
    page.goto(
        "https://github.com/Swathi-A01/heart-disease-mlops",
        wait_until="domcontentloaded",
        timeout=30000
    )
    time.sleep(3)
    shot(page, "15_github_repo_home")

    # ── 3. Prometheus ─────────────────────────────────────────────────────
    print("Prometheus...")
    page.goto("http://localhost:9090/targets", wait_until="domcontentloaded")
    time.sleep(3)
    shot(page, "16_prometheus_targets")

    page.goto("http://localhost:9090/graph", wait_until="domcontentloaded")
    time.sleep(2)
    try:
        page.fill("textarea.cm-content, .cm-content, input[type=text]",
                  "heart_risk_predictions_total")
    except Exception:
        try:
            page.keyboard.type("heart_risk_predictions_total")
        except Exception:
            pass
    page.keyboard.press("Enter")
    time.sleep(2)
    shot(page, "17_prometheus_query")

    # ── 4. Grafana ────────────────────────────────────────────────────────
    print("Grafana...")
    page.goto("http://localhost:3000/login", wait_until="domcontentloaded")
    time.sleep(2)
    try:
        page.fill("input[name='user']", "admin")
        page.fill("input[name='password']", "admin")
        page.click("button[type='submit']")
        time.sleep(3)
        try:
            skip = page.query_selector("text=Skip")
            if skip:
                skip.click()
                time.sleep(1)
        except Exception:
            pass
    except Exception as e:
        print(f"  Grafana login: {e}")

    page.goto(
        "http://localhost:3000/d/heart-risk-dashboard/heart-disease-risk-api",
        wait_until="domcontentloaded"
    )
    time.sleep(5)
    shot(page, "18_grafana_dashboard")

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.reload(wait_until="domcontentloaded")
    time.sleep(5)
    shot(page, "19_grafana_dashboard_wide")

    # ── 5. API predict JSON responses ─────────────────────────────────────
    print("API predict responses...")
    SAMPLE_HIGH = {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
        "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7
    }
    SAMPLE_LOW = {
        "age": 45, "sex": 0, "cp": 1, "trestbps": 110, "chol": 190,
        "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    }

    for label, sample, fname in [
        ("High Risk", SAMPLE_HIGH, "20_api_predict_high_risk"),
        ("Low Risk", SAMPLE_LOW, "21_api_predict_low_risk"),
    ]:
        req = urllib.request.Request(
            "http://localhost:8000/predict",
            data=json.dumps(sample).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp_raw = urllib.request.urlopen(req).read().decode()
        resp_pretty = json.dumps(json.loads(resp_raw), indent=2)
        # Render as a clean page
        html = f"""<html><body style="background:#1e1e2e;color:#cdd6f4;font-family:monospace;
            font-size:16px;padding:30px;">
            <h2 style="color:#89b4fa;">POST /predict — {label} Patient</h2>
            <h3 style="color:#a6e3a1;">Request:</h3>
            <pre style="background:#313244;padding:15px;border-radius:8px;color:#cdd6f4">{json.dumps(sample, indent=2)}</pre>
            <h3 style="color:#a6e3a1;">Response:</h3>
            <pre style="background:#313244;padding:15px;border-radius:8px;color:#a6e3a1">{resp_pretty}</pre>
            </body></html>"""
        page.set_content(html)
        shot(page, fname)

    # Batch predict
    req_b = urllib.request.Request(
        "http://localhost:8000/predict-batch",
        data=json.dumps([SAMPLE_HIGH, SAMPLE_LOW, SAMPLE_HIGH]).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    resp_b = json.loads(urllib.request.urlopen(req_b).read().decode())
    resp_b_summary = {
        "count": resp_b["count"],
        "high_risk_count": resp_b["high_risk_count"],
        "low_risk_count": resp_b["low_risk_count"],
        "results": [{"prediction": r["prediction"], "probability": r["probability"],
                     "risk": r["risk"]} for r in resp_b["results"]]
    }
    html_b = f"""<html><body style="background:#1e1e2e;color:#cdd6f4;font-family:monospace;
        font-size:15px;padding:30px;">
        <h2 style="color:#89b4fa;">POST /predict-batch — 3 Patients</h2>
        <pre style="background:#313244;padding:15px;border-radius:8px;color:#a6e3a1">{json.dumps(resp_b_summary, indent=2)}</pre>
        </body></html>"""
    page.set_content(html_b)
    shot(page, "22_api_predict_batch")

    browser.close()

shots_list = sorted(SHOTS.glob("*.png"))
print(f"\nAll done — {len(shots_list)} screenshots in {SHOTS.resolve()}")
for s in shots_list:
    print(f"  {s.name}")
