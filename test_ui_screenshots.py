"""Visual smoke test: renders every view in a real (headless) browser and saves screenshots.

Run locally:  pip install playwright && playwright install chromium && pytest tests/test_ui_screenshots.py
Skipped automatically when Playwright or a browser is not available. Screenshots go to results/screenshots/.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "results" / "screenshots"
VIEWS = ["Home", "Diagnostics", "Live twin", "Models & forecasting", "Operations & control", "Study results"]
DIAG_TABS = ["Overview", "Ageing", "Electrochemistry", "Stress & safety"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless=true",
                             f"--server.port={port}"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(1)
    yield url
    proc.terminate()


def _no_errors(page) -> None:
    txt = page.inner_text("body")
    assert "Traceback" not in txt and "failed:" not in txt, txt[:500]


def test_every_view_renders(app_url):
    SHOTS.mkdir(parents=True, exist_ok=True)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                         # no browser installed
            pytest.skip(f"chromium not available: {exc}")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(app_url, wait_until="networkidle")
        page.get_by_role("button", name="Load synthetic demo").click()
        page.wait_for_timeout(8000)
        for v in VIEWS:
            page.get_by_text(v, exact=False).first.click()
            page.wait_for_timeout(6000)
            _no_errors(page)
            page.screenshot(path=str(SHOTS / f"{v.split()[0].lower()}.png"), full_page=True)
            if v == "Diagnostics":
                for t in DIAG_TABS[1:]:
                    page.get_by_text(t, exact=True).first.click()
                    page.wait_for_timeout(6000)
                    _no_errors(page)
                    page.screenshot(path=str(SHOTS / f"diagnostics_{t.split()[0].lower()}.png"), full_page=True)
        for width in (1024, 760):                         # narrow screens
            page.set_viewport_size({"width": width, "height": 900})
            page.get_by_text("Home", exact=False).first.click()
            page.wait_for_timeout(4000)
            page.screenshot(path=str(SHOTS / f"home_{width}px.png"), full_page=True)
        browser.close()
