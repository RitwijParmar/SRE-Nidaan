#!/usr/bin/env python3
"""
Record a product demonstration video from the live SRE Nidaan site.

Output:
  presentations/SRE_Nidaan_Demo_Recording.webm
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PRODUCT_URL = "https://sre-nidaan-122722888597.us-east4.run.app"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentations"
VIDEO_DIR = OUT_DIR / "_raw_recordings"
FINAL_VIDEO = OUT_DIR / "SRE_Nidaan_Demo_Recording.webm"


def pause(seconds: float) -> None:
    time.sleep(seconds)


def click_if_visible(page, selector: str, timeout_ms: int = 4000) -> bool:
    try:
        loc = page.locator(selector)
        loc.wait_for(timeout=timeout_ms, state="visible")
        loc.click()
        return True
    except PlaywrightTimeoutError:
        return False


def fill_if_visible(page, selector: str, value: str, timeout_ms: int = 4000) -> bool:
    try:
        loc = page.locator(selector)
        loc.wait_for(timeout=timeout_ms, state="visible")
        loc.fill(value)
        return True
    except PlaywrightTimeoutError:
        return False


def run_demo() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()

        page.goto(PRODUCT_URL, wait_until="networkidle", timeout=60000)
        pause(1.2)

        # Step 1: sanity check runtime
        click_if_visible(page, "button:has-text('Check System')")
        pause(2.0)

        # Step 2: load a realistic incident preset
        click_if_visible(page, "button:has-text('Auth Retry Storm')")
        pause(1.2)

        # Step 3: refine final brief slightly to show operator input
        fill_if_visible(
            page,
            "textarea[placeholder='Generate from the fields above, then edit this final brief before analysis.']",
            (
                "SEV-1 auth incident: p95 latency climbed from 210ms to 1.8s, "
                "error rate rose to 18%, and DB connections reached 99%. "
                "Login and checkout are timing out across EU/US. "
                "Auth middleware was rolled out 18 minutes before spike. "
                "Suspected retry storm on auth-service and api-gateway against postgres."
            ),
        )
        pause(1.4)

        # Step 4: run analysis
        if click_if_visible(page, "#analyze-incident-btn", timeout_ms=7000):
            pause(3.0)
            try:
                page.locator("text=Analysis complete.").wait_for(timeout=95000)
            except PlaywrightTimeoutError:
                # Continue demo even if analysis times out/fails.
                pass

        # Step 5: keep graph area visible
        page.locator("#causal-graph-workspace").scroll_into_view_if_needed()
        pause(4.0)

        # Step 6: show safety section
        if page.locator("text=4. Safety Approval").count() > 0:
            page.locator("text=4. Safety Approval").first.scroll_into_view_if_needed()
            pause(1.0)
            fill_if_visible(
                page,
                "textarea[placeholder='Why is this safe to approve?']",
                "Manual rollback gate in place, limited blast radius, and on-call DBA notified.",
                timeout_ms=6000,
            )
            pause(0.8)
            click_if_visible(page, "#authorize-intervention-btn", timeout_ms=4000)
            pause(1.8)

        # Step 7: show feedback loop
        if page.locator("text=5. Analyst Feedback").count() > 0:
            page.locator("text=5. Analyst Feedback").first.scroll_into_view_if_needed()
            pause(0.8)
            fill_if_visible(
                page,
                "textarea[placeholder='Optional correction or missing context...']",
                "Useful output. Recommend adding explicit DB pool saturation trend in root cause summary.",
                timeout_ms=5000,
            )
            pause(0.8)
            click_if_visible(page, "button:has-text('Mark Useful')", timeout_ms=3000)
            pause(1.5)

        # End hold for clean outro frame
        page.locator("header").first.scroll_into_view_if_needed()
        pause(2.2)

        video_path = Path(page.video.path())
        context.close()
        browser.close()

    return video_path


def main() -> None:
    raw_video = run_demo()
    if FINAL_VIDEO.exists():
        FINAL_VIDEO.unlink()
    shutil.copy(raw_video, FINAL_VIDEO)
    print(f"Recorded demo: {FINAL_VIDEO}")


if __name__ == "__main__":
    main()

