#!/usr/bin/env python3
"""
Generate SRE Nidaan demo assets:
1) PPT deck for presentations
2) LinkedIn demo video script + shotlist
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentations"
PPT_PATH = OUT_DIR / "SRE_Nidaan_Demo_Deck.pptx"


COLORS = {
    "bg": RGBColor(245, 249, 255),
    "ink": RGBColor(20, 36, 58),
    "muted": RGBColor(77, 96, 122),
    "accent": RGBColor(10, 127, 120),
    "accent2": RGBColor(33, 102, 207),
    "border": RGBColor(194, 211, 232),
}


def style_text(tf, size=22, bold=False, color="ink"):
    if not tf.paragraphs:
        tf.text = ""
    for paragraph in tf.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = COLORS[color]


def add_bg(slide):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(0),
        Inches(0),
        Inches(13.333),
        Inches(7.5),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLORS["bg"]
    shape.line.fill.background()


def add_title(slide, title: str, subtitle: str = ""):
    box = slide.shapes.add_textbox(Inches(0.7), Inches(0.5), Inches(11.9), Inches(1.5))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    style_text(tf, size=36, bold=True, color="ink")

    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.75), Inches(1.55), Inches(11.8), Inches(1.2))
        stf = sub.text_frame
        stf.clear()
        sp = stf.paragraphs[0]
        sp.text = subtitle
        style_text(stf, size=18, bold=False, color="muted")


def bullet_block(slide, heading: str, bullets: list[str], top: float):
    head = slide.shapes.add_textbox(Inches(0.9), Inches(top), Inches(11.7), Inches(0.6))
    htf = head.text_frame
    htf.text = heading
    style_text(htf, size=24, bold=True, color="accent2")

    box = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(0.8),
        Inches(top + 0.55),
        Inches(11.8),
        Inches(2.3),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(255, 255, 255)
    box.line.color.rgb = COLORS["border"]

    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
    style_text(tf, size=18, bold=False, color="ink")


def architecture_slide(slide):
    add_title(slide, "Architecture", "Face → Body → Brain, with a safety-first control plane")

    y = Inches(2.1)
    h = Inches(2.0)
    w = Inches(3.3)
    x1, x2, x3 = Inches(0.9), Inches(4.95), Inches(8.95)
    labels = [
        ("Face", "Next.js command surface\nOperator workflow"),
        ("Body", "FastAPI orchestration\nMCP-inspired tool routing\nPolicy + safety gating"),
        ("Brain", "vLLM inference service\nLLM reasoning backend"),
    ]
    for x, (name, desc) in zip([x1, x2, x3], labels):
        box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, x, y, w, h)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(255, 255, 255)
        box.line.color.rgb = COLORS["border"]

        title = slide.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), w - Inches(0.4), Inches(0.5))
        ttf = title.text_frame
        ttf.text = name
        style_text(ttf, size=22, bold=True, color="accent")

        txt = slide.shapes.add_textbox(x + Inches(0.2), y + Inches(0.75), w - Inches(0.4), Inches(1.2))
        tf = txt.text_frame
        tf.text = desc
        style_text(tf, size=14, bold=False, color="ink")

    for left in [Inches(4.3), Inches(8.35)]:
        arr = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW, left, Inches(2.75), Inches(0.45), Inches(0.45))
        arr.fill.solid()
        arr.fill.fore_color.rgb = COLORS["accent2"]
        arr.line.fill.background()

    note = slide.shapes.add_textbox(Inches(0.95), Inches(5.05), Inches(11.4), Inches(1.1))
    ntf = note.text_frame
    ntf.text = (
        "Why this helps: split responsibilities keep incident workflows clear, auditable, and safer under pressure."
    )
    style_text(ntf, size=16, bold=False, color="muted")


def make_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(
        s,
        "SRE निदान (SRE-Nidaan)",
        "Causal incident response copilot for reliability teams",
    )
    bullet_block(
        s,
        "One-line intent",
        [
            "When incidents get noisy, help teams answer:",
            "• What broke first?",
            "• What should we do now?",
            "• What should we definitely not touch?",
        ],
        top=2.35,
    )

    # Slide 2
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Problem", "Why standard LLM behavior is risky in incidents")
    bullet_block(
        s,
        "Observed gaps",
        [
            "• Fluent responses can still be operationally wrong.",
            "• Generic advice ignores blast radius and change context.",
            "• Unstructured output slows triage and handoff.",
            "• No safety gate means risky actions can be suggested too casually.",
        ],
        top=2.1,
    )

    # Slide 3
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Approach", "Practical copilot, not autopilot")
    bullet_block(
        s,
        "Design principles",
        [
            "• Grounded reasoning from telemetry + incident context.",
            "• Structured outputs for predictable operator consumption.",
            "• Human approval gate for intervention authorization.",
            "• Operator feedback loop for continuous model improvement.",
        ],
        top=2.1,
    )

    # Slide 4
    s = prs.slides.add_slide(blank)
    add_bg(s)
    architecture_slide(s)

    # Slide 5
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Model Pipeline", "LLM alignment stack used in SRE Nidaan")
    bullet_block(
        s,
        "Training sequence",
        [
            "• QLoRA Supervised Fine-Tuning (domain incident data).",
            "• 7D reward modeling (safety + causality dimensions).",
            "• RLHF refinement with guardrails and periodic eval gating.",
            "• Runtime serving through vLLM for low-latency inference.",
        ],
        top=2.1,
    )

    # Slide 6
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Live Workflow", "How an incident run looks end-to-end")
    bullet_block(
        s,
        "Operator flow",
        [
            "1. Describe incident signal/impact/change context.",
            "2. Run analysis and inspect causal graph + evidence.",
            "3. Review recommended intervention and safety check.",
            "4. Authorize manually, then submit analyst feedback.",
        ],
        top=2.1,
    )

    # Slide 7
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Demo Script", "Suggested 90-second walkthrough")
    bullet_block(
        s,
        "Timeline",
        [
            "• 0:00-0:20: Incident framing and objective.",
            "• 0:20-0:45: Fill incident brief, run analysis.",
            "• 0:45-1:10: Inspect graph + intervention + safety gate.",
            "• 1:10-1:30: Feedback loop + close with links.",
        ],
        top=2.1,
    )

    # Slide 8
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Why Useful", "What this improves during high-pressure incidents")
    bullet_block(
        s,
        "Operational value",
        [
            "• Faster first-pass causal clarity.",
            "• Better consistency across responders.",
            "• Safer interventions through explicit approval gates.",
            "• Cleaner post-incident learning via structured feedback.",
        ],
        top=2.1,
    )

    # Slide 9
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Current State", "Honest status")
    bullet_block(
        s,
        "Reality check",
        [
            "• Early, imperfect but seems useful in practice.",
            "• Focus remains production hardening, not hype.",
            "• Goal: reliable copilot behavior under real incident pressure.",
        ],
        top=2.3,
    )

    # Slide 10
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "Links", "Try it and review the code")
    bullet_block(
        s,
        "Public endpoints",
        [
            "Product: https://sre-nidaan-122722888597.us-east4.run.app",
            "GitHub: https://github.com/RitwijParmar/SRE-Nidaan",
        ],
        top=2.35,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(PPT_PATH)


def make_linkedin_files():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "linkedin_demo_video_script.md").write_text(
        """# SRE Nidaan LinkedIn Demo Script (90 sec)

## Voiceover
When incidents hit, teams usually ask three questions: what broke first, what should we do now, and what should we definitely not touch.

I built SRE Nidaan to make that reasoning faster and safer.

This system runs as three services: a command interface, an orchestration and safety layer, and an LLM inference layer.

The model stack uses QLoRA SFT, reward modeling, and RLHF, with an MCP-inspired tool interface so incident reasoning can pull structured telemetry through controlled calls.

Here is a live run: we describe the incident context, run analysis, inspect the causal graph, review intervention logic, and keep human approval mandatory before actions.

The goal is not autopilot. The goal is a practical copilot that helps teams think clearly under pressure.

Early, imperfect but seems useful in practice.

Product and code are linked below.
""",
        encoding="utf-8",
    )

    (OUT_DIR / "linkedin_demo_shotlist.md").write_text(
        """# SRE Nidaan LinkedIn Shotlist (90 sec)

## 0:00 - 0:10
- Show product home and title.
- On-screen text: "SRE निदान: Incident reasoning copilot"

## 0:10 - 0:25
- Fill incident fields quickly (signal, impact, change context).
- On-screen text: "Context in. Noisy incident -> structured brief."

## 0:25 - 0:40
- Click Analyze Incident.
- Keep loading + runtime status visible.

## 0:40 - 1:00
- Show causal graph + graph inspector.
- Highlight root cause and intervention logic.

## 1:00 - 1:15
- Show safety approval section (manual authorization requirement).
- On-screen text: "Human approval stays in the loop."

## 1:15 - 1:30
- Show feedback section and final screen with links.
- On-screen text:
  - Product URL
  - GitHub URL
""",
        encoding="utf-8",
    )

    (OUT_DIR / "linkedin_caption.txt").write_text(
        """I’ve been working on SRE निदान (SRE-Nidaan) for that incident moment when everyone asks:
What broke first? What should we do now? What should we definitely not touch?

This is not an “AI solves SRE” claim. It’s a practical copilot approach for clearer, safer reasoning under pressure.

Under the hood: a 3-service architecture (inference + orchestration + safety gating), with a model pipeline using QLoRA SFT, reward modeling, and RLHF, plus an MCP-inspired tool layer for controlled telemetry calls.

Early, imperfect but seems useful in practice.

Product: https://sre-nidaan-122722888597.us-east4.run.app
GitHub: https://github.com/RitwijParmar/SRE-Nidaan
""",
        encoding="utf-8",
    )

    (OUT_DIR / "README_demo_assets.md").write_text(
        """# Demo Assets

Generated files:
- `SRE_Nidaan_Demo_Deck.pptx`
- `linkedin_demo_video_script.md`
- `linkedin_demo_shotlist.md`
- `linkedin_caption.txt`

## Suggested flow
1. Open the product URL in browser.
2. Follow the shotlist and read from script.
3. Keep video between 75 and 90 seconds.
4. Post with the provided caption text.
""",
        encoding="utf-8",
    )


def main():
    make_deck()
    make_linkedin_files()
    print(f"Created demo assets in: {OUT_DIR}")


if __name__ == "__main__":
    main()

