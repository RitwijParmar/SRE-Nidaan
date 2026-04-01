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
ANALYSIS_TIMEOUT_SECONDS = 105


def pause(seconds: float) -> None:
    time.sleep(seconds)


def remove_spotlight(page) -> None:
    page.evaluate(
        """
        () => {
          const el = document.getElementById("__demo_spotlight");
          if (el) {
            el.remove();
          }
        }
        """
    )


def spotlight_locator(page, locator, label: str, hold_seconds: float = 1.4) -> None:
    box = locator.bounding_box()
    if not box:
        return
    page.evaluate(
        """
        ({ x, y, width, height, label }) => {
          const old = document.getElementById("__demo_spotlight");
          if (old) {
            old.remove();
          }

          const root = document.createElement("div");
          root.id = "__demo_spotlight";
          root.style.position = "fixed";
          root.style.inset = "0";
          root.style.pointerEvents = "none";
          root.style.zIndex = "2147483647";

          const ring = document.createElement("div");
          ring.style.position = "fixed";
          ring.style.left = `${Math.max(8, x - 8)}px`;
          ring.style.top = `${Math.max(8, y - 8)}px`;
          ring.style.width = `${Math.max(24, width + 16)}px`;
          ring.style.height = `${Math.max(24, height + 16)}px`;
          ring.style.border = "3px solid rgba(229, 57, 53, 0.92)";
          ring.style.borderRadius = "14px";
          ring.style.boxShadow = "0 0 0 9999px rgba(17, 24, 39, 0.14)";
          ring.style.animation = "demoPulse 1.1s ease-in-out infinite";

          const bubble = document.createElement("div");
          bubble.textContent = label;
          bubble.style.position = "fixed";
          bubble.style.left = `${Math.max(12, x)}px`;
          bubble.style.top = `${Math.max(12, y - 42)}px`;
          bubble.style.padding = "6px 10px";
          bubble.style.background = "rgba(13, 31, 54, 0.95)";
          bubble.style.color = "white";
          bubble.style.font = "600 12px/1.2 system-ui, -apple-system, Segoe UI, sans-serif";
          bubble.style.borderRadius = "999px";
          bubble.style.letterSpacing = "0.01em";

          const style = document.createElement("style");
          style.textContent = `
            @keyframes demoPulse {
              0% { transform: scale(1); opacity: 1; }
              50% { transform: scale(1.018); opacity: 0.78; }
              100% { transform: scale(1); opacity: 1; }
            }
          `;

          root.appendChild(style);
          root.appendChild(ring);
          root.appendChild(bubble);
          document.body.appendChild(root);
        }
        """,
        {
            "x": box["x"],
            "y": box["y"],
            "width": box["width"],
            "height": box["height"],
            "label": label,
        },
    )
    pause(hold_seconds)
    remove_spotlight(page)


def scroll_to_locator(locator) -> None:
    locator.evaluate("el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
    pause(1.3)


def click_if_visible(page, selector: str, timeout_ms: int = 4000) -> bool:
    try:
        loc = page.locator(selector).first
        loc.wait_for(timeout=timeout_ms, state="visible")
        scroll_to_locator(loc)
        spotlight_locator(page, loc, "Click here", hold_seconds=1.0)
        box = loc.bounding_box()
        if box:
            page.mouse.move(
                max(10, box["x"] - 80),
                min(860, box["y"] + box["height"] + 40),
                steps=14,
            )
            pause(0.18)
            page.mouse.move(
                box["x"] + box["width"] / 2,
                box["y"] + box["height"] / 2,
                steps=20,
            )
            pause(0.08)
        loc.click()
        return True
    except PlaywrightTimeoutError:
        return False


def fill_if_visible(page, selector: str, value: str, timeout_ms: int = 4000) -> bool:
    try:
        loc = page.locator(selector).first
        loc.wait_for(timeout=timeout_ms, state="visible")
        scroll_to_locator(loc)
        spotlight_locator(page, loc, "Update incident details", hold_seconds=1.0)
        loc.click()
        loc.fill("")
        loc.type(value, delay=8)
        return True
    except PlaywrightTimeoutError:
        return False


def wait_for_analysis_state(page) -> str:
    deadline = time.time() + ANALYSIS_TIMEOUT_SECONDS
    while time.time() < deadline:
        if page.locator("#authorize-intervention-btn").count() > 0:
            return "success"
        if page.locator("text=Live analysis failed").count() > 0:
            return "failed"
        if page.locator("text=Synthetic fallback rendering is disabled in product mode.").count() > 0:
            return "failed"
        pause(1.2)
    return "timeout"


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
        page.locator("h1:has-text('SRE निदान')").first.wait_for(timeout=20000)
        pause(1.0)

        # Step 1: sanity check runtime
        click_if_visible(page, "button:has-text('Check System')", timeout_ms=7000)
        pause(2.2)

        # Step 2: load a realistic incident preset
        click_if_visible(page, "button:has-text('Auth Retry Storm')", timeout_ms=7000)
        pause(1.4)

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
            timeout_ms=9000,
        )
        pause(1.0)

        # Step 4: run analysis
        analyze_button = page.locator("#analyze-incident-btn").first
        scroll_to_locator(analyze_button)
        spotlight_locator(page, analyze_button, "Run analysis", hold_seconds=1.2)
        click_if_visible(page, "#analyze-incident-btn", timeout_ms=7000)
        pause(2.8)
        analysis_state = wait_for_analysis_state(page)

        # Step 5: show graph area
        graph_workspace = page.locator("#causal-graph-workspace").first
        scroll_to_locator(graph_workspace)
        spotlight_locator(page, graph_workspace, "Inspect causal graph", hold_seconds=2.6)
        pause(1.4)

        if analysis_state != "success":
            # Keep the run useful even if backend inference did not complete.
            runtime_card = page.locator("article:has-text('2. Runtime Readiness')").first
            scroll_to_locator(runtime_card)
            spotlight_locator(page, runtime_card, "Runtime diagnostics", hold_seconds=2.1)
            pause(1.5)

        # Step 6: show safety section if analysis succeeded
        if analysis_state == "success" and page.locator("text=4. Safety Approval").count() > 0:
            safety_section = page.locator("text=4. Safety Approval").first
            scroll_to_locator(safety_section)
            spotlight_locator(page, safety_section, "Safety gate before action", hold_seconds=1.7)
            fill_if_visible(
                page,
                "textarea[placeholder='Why is this safe to approve?']",
                "Manual rollback gate in place, limited blast radius, and on-call DBA notified.",
                timeout_ms=6000,
            )
            pause(0.6)
            click_if_visible(page, "#authorize-intervention-btn", timeout_ms=4000)
            pause(2.0)

        # Step 7: show feedback loop if analysis succeeded
        if analysis_state == "success" and page.locator("text=5. Analyst Feedback").count() > 0:
            feedback_section = page.locator("text=5. Analyst Feedback").first
            scroll_to_locator(feedback_section)
            spotlight_locator(page, feedback_section, "Close feedback loop", hold_seconds=1.8)
            fill_if_visible(
                page,
                "textarea[placeholder='Optional correction or missing context...']",
                "Useful output. Recommend adding explicit DB pool saturation trend in root cause summary.",
                timeout_ms=5000,
            )
            pause(0.6)
            click_if_visible(page, "button:has-text('Mark Useful')", timeout_ms=3000)
            pause(2.2)

        # End hold for clean outro frame
        header = page.locator("header").first
        scroll_to_locator(header)
        logo = page.locator("h1:has-text('SRE निदान')").first
        spotlight_locator(page, logo, "SRE Nidaan workflow complete", hold_seconds=1.8)
        pause(1.8)

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
