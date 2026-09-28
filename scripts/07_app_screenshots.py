"""Capture screenshots of the running viewer app for the report.

Start the app first:  streamlit run app/streamlit_app.py --server.port 8531
Needs: pip install playwright (uses the installed Google Chrome).
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "app"
SHOTS = [
    ("app_nagpur_2019.png", "region=Nagpur&date=2019-05-31"),
    ("app_mumbai_2024.png", "region=Mumbai&date=2024-05-10"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1250, "height": 1400}, device_scale_factor=2)
        for name, query in SHOTS:
            page.goto(f"http://localhost:8531/?{query}", wait_until="networkidle")
            page.wait_for_selector("[data-testid='stDataFrame']", timeout=90000)
            page.wait_for_timeout(4000)
            page.add_style_tag(content="header, [data-testid='stToolbar'], "
                                       "[data-testid='stDecoration'] {display:none !important}")
            page.locator("[data-testid='stMainBlockContainer']").screenshot(path=str(OUT / name))
            print("saved", OUT / name)
        browser.close()
