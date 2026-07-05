"""
theme.py
Loads the custom frontend/ CSS assets into the Streamlit app, and
provides helpers for building the themed HTML components used in
streamlit_app.py (hero header, scan-frame images, signal meters,
about section, structured report cards).

IMPORTANT: every HTML string rendered through st.markdown() is passed
through `_html()` first, which strips leading whitespace from each
line. Streamlit's markdown renderer follows CommonMark, where any line
indented 4+ spaces is treated as a code block -- multi-line f-strings
built inside indented Python code trip this constantly, causing raw
HTML tags to show up as visible text instead of rendering. Stripping
indentation before rendering avoids that class of bug everywhere,
rather than fixing it call-by-call.
"""

import base64
import io
import random
import re

import streamlit as st
from PIL import Image

from python.utils.config_loader import resolve_path


def _html(raw: str) -> str:
    """Strips leading/trailing whitespace from every line of a
    multi-line HTML string so Streamlit's markdown parser never
    mistakes indented HTML for a fenced code block."""
    return "\n".join(line.strip() for line in raw.strip().splitlines())


def render_html(raw: str):
    """Shortcut for st.markdown(_html(raw), unsafe_allow_html=True)."""
    st.markdown(_html(raw), unsafe_allow_html=True)


def inject_theme():
    """Loads frontend/css/theme.css into the page via a <style> block,
    and renders an ambient particle field directly in the app's own DOM
    (not inside a components.html iframe, which clips position:fixed
    content to the iframe's own box rather than the real viewport)."""
    css_path = resolve_path("frontend/css/theme.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _render_ambient_particles()


def _render_ambient_particles(count: int = 34):
    """Pure CSS/HTML ambient particle field: small glowing dots drifting
    slowly with randomized position/timing, rendered as a fixed
    full-viewport layer behind all content. No JS/canvas -- this lives
    in the real page DOM so position:fixed actually covers the screen."""
    random.seed(7)  # stable layout across reruns instead of reshuffling every interaction
    dots = []
    for _ in range(count):
        top = random.uniform(0, 100)
        left = random.uniform(0, 100)
        size = random.uniform(1.5, 3.2)
        duration = random.uniform(14, 30)
        delay = random.uniform(0, 10)
        opacity = random.uniform(0.25, 0.6)
        dots.append(
            f'<div class="mv-particle" style="top:{top}%; left:{left}%; '
            f'width:{size}px; height:{size}px; '
            f'animation-duration:{duration}s; animation-delay:-{delay}s; '
            f'--mv-p-opacity:{opacity};"></div>'
        )
    render_html(f'<div class="mv-particle-field">{"".join(dots)}</div>')


def render_header(title: str, subtitle: str, status_label: str = "SYSTEM ONLINE"):
    """Renders the glass-panel hero header with a typewriter-animated
    title and subtitle, logo, and a live status pill."""
    render_html(
        f"""
        <div class="mv-header">
            <div class="mv-logo-icon">🫁</div>
            <div>
                <p class="mv-title mv-typewriter-title">{title}</p>
                <p class="mv-subtitle mv-typewriter-sub">{subtitle}</p>
            </div>
            <div class="mv-status-pill">
                <span class="mv-status-dot"></span>{status_label}
            </div>
        </div>
        """
    )


def render_about_section():
    """Renders a 3-card 'what / how / why' about section explaining the
    project's purpose and build, so a first-time visitor understands
    what they're looking at before uploading anything."""
    render_html(
        """
        <div class="mv-about-grid">
            <div class="mv-about-card">
                <div class="mv-about-icon">\U0001FA7B</div>
                <div class="mv-about-title">What it does</div>
                <div class="mv-about-body">
                    Upload a chest X-ray and get an AI-assisted read: a multi-label
                    classifier flags likely findings, a visual explainability layer
                    shows exactly where it looked, and a language model drafts a
                    structured preliminary report grounded in real clinical
                    reference material.
                </div>
            </div>
            <div class="mv-about-card">
                <div class="mv-about-icon">\u2699\ufe0f</div>
                <div class="mv-about-title">How it's built</div>
                <div class="mv-about-body">
                    A ResNet-50 CNN fine-tuned on NIH ChestX-ray14 for 14-condition
                    detection, Grad-CAM for explainability, a ChromaDB + sentence-
                    transformers retrieval layer grounding responses in clinical
                    guidance text, and a locally-run Ollama (Mistral-7B) model for
                    report drafting. No paid APIs anywhere in the pipeline.
                </div>
            </div>
            <div class="mv-about-card">
                <div class="mv-about-icon">\U0001F50D</div>
                <div class="mv-about-title">Why it's explainable</div>
                <div class="mv-about-body">
                    Every prediction ships with a Grad-CAM heatmap showing the
                    region that drove it, and every drafted report cites the
                    specific reference chunks it was grounded in -- nothing here
                    is a black box, and nothing replaces physician review.
                </div>
            </div>
        </div>
        """
    )


def render_pipeline_chips(steps: list):
    """Renders the pipeline steps as glowing chips instead of a plain
    numbered markdown list."""
    chips_html = "".join(
        f'<div class="mv-pipeline-chip">'
        f'<span class="mv-pipeline-num">{i+1:02d}</span>{step}</div>'
        for i, step in enumerate(steps)
    )
    render_html(f'<div class="mv-pipeline">{chips_html}</div>')


def _image_to_base64(image) -> str:
    """Accepts a PIL Image or numpy array and returns a base64 PNG data URI."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def render_scan_frame(image, label: str):
    """Wraps an image in the signature HUD scan-frame: glowing corner
    brackets, an animated sweep line, and a monospace caption label."""
    data_uri = _image_to_base64(image)
    render_html(
        f"""
        <div class="mv-scan-frame">
            <div class="mv-corner tl"></div>
            <div class="mv-corner tr"></div>
            <div class="mv-corner bl"></div>
            <div class="mv-corner br"></div>
            <div class="mv-scanline"></div>
            <img src="{data_uri}" />
            <div class="mv-frame-label">{label}</div>
        </div>
        """
    )


def render_signal_meters(findings: dict, threshold: float):
    """Renders classifier findings as glowing horizontal signal-meter
    bars (amber = flagged above threshold, dim cyan = below threshold),
    with monospace percentage readouts -- replaces the plain bar chart."""
    rows_html = []
    for name, prob in findings.items():
        flagged = prob >= threshold
        state = "flagged" if flagged else "clear"
        pct = prob * 100
        rows_html.append(
            f"""
            <div class="mv-meter-row">
                <div class="mv-meter-label">{name}</div>
                <div class="mv-meter-track">
                    <div class="mv-meter-fill {state}" style="width:{pct:.1f}%;"></div>
                </div>
                <div class="mv-meter-value {state}">{pct:5.1f}%</div>
            </div>
            """
        )
    render_html("".join(rows_html))


def render_eyebrow(text: str):
    """Small monospace uppercase label used above section headings."""
    render_html(f'<div class="mv-eyebrow">{text}</div>')


def render_hero_scan(model_info: dict, dataset_info: dict, pipeline_steps: list, status_rows: list):
    """Renders the hero section: a central rotating holographic scan
    core flanked by two HUD data panels of real project data (model
    specs, dataset stats, live pipeline stages, system status)."""
    model_rows = "".join(
        f'<div class="mv-hud-row"><span>{k}</span><span>{v}</span></div>'
        for k, v in model_info.items()
    )
    dataset_rows = "".join(
        f'<div class="mv-hud-row"><span>{k}</span><span>{v}</span></div>'
        for k, v in dataset_info.items()
    )
    pipeline_rows = "".join(
        f'<div class="mv-hud-row"><span>{i+1:02d} {step}</span><span></span></div>'
        for i, step in enumerate(pipeline_steps)
    )
    status_html = "".join(
        f'<div class="mv-hud-row {css_class}"><span>{k}</span><span>{v}</span></div>'
        for k, v, css_class in status_rows
    )

    render_html(
        f"""
        <div class="mv-hero">
            <div class="mv-hero-col">
                <div class="mv-hud-panel">
                    <div class="mv-hud-title">Model</div>
                    {model_rows}
                </div>
                <div class="mv-hud-panel">
                    <div class="mv-hud-title">Dataset</div>
                    {dataset_rows}
                </div>
            </div>
            <div>
                <div class="mv-scan-core">
                    <div class="mv-scan-ring"></div>
                    <div class="mv-scan-ring r2"></div>
                    <div class="mv-scan-ring r3"></div>
                    <div class="mv-scan-pulse"></div>
                    <div class="mv-scan-node" style="top:6px; left:50%;"></div>
                    <div class="mv-scan-node" style="bottom:14px; right:22px;"></div>
                    <div class="mv-scan-node" style="bottom:20px; left:18px;"></div>
                    <div class="mv-scan-icon">\U0001FAC1</div>
                </div>
                <div class="mv-scan-caption">Diagnostic Core // Standby</div>
            </div>
            <div class="mv-hero-col">
                <div class="mv-hud-panel">
                    <div class="mv-hud-title">Pipeline</div>
                    {pipeline_rows}
                </div>
                <div class="mv-hud-panel">
                    <div class="mv-hud-title">System Status</div>
                    {status_html}
                </div>
            </div>
        </div>
        """
    )


def control_label(text: str):
    render_html(f'<span class="mv-control-label">{text}</span>')


def render_floating_nav(sections: list):
    """Renders a fixed, floating glass navbar with smooth-scroll anchor
    links to each pipeline stage. `sections` is a list of (id, label)
    tuples, e.g. [("sec-input", "Input")]."""
    links_html = "".join(
        f'<a class="mv-navlink" href="#{anchor_id}">'
        f'<span class="mv-navlink-num">{i+1:02d}</span>{label}</a>'
        for i, (anchor_id, label) in enumerate(sections)
    )
    render_html(
        f"""
        <div class="mv-navbar">
            <div class="mv-navbar-brand">\U0001FAC1 MEDIVISION</div>
            <div class="mv-navbar-links">{links_html}</div>
        </div>
        <div class="mv-nav-spacer"></div>
        """
    )


def render_anchor(anchor_id: str):
    """Drops an invisible scroll-target anchor above a section heading
    so the floating navbar's links can jump to it, offset so the
    heading isn't hidden behind the fixed navbar."""
    render_html(f'<div id="{anchor_id}" class="mv-anchor"></div>')


def _parse_guidance_text(text: str):
    """Splits a retrieved guidance chunk into an intro paragraph and a
    list of 'suggested next step' bullets. See build_index.py's
    chunk_text(): newlines collapse to single spaces, so bullets
    survive only as ' - Item' markers -- this reconstructs them into a
    real list. Falls back to the whole chunk as intro if not found."""
    match = re.split(r"Suggested next steps[^:]*:", text, maxsplit=1, flags=re.IGNORECASE)
    intro = match[0].strip()

    steps = []
    if len(match) > 1:
        raw_items = re.split(r"\s-\s(?=[A-Z])", match[1].strip().lstrip("- ").strip())
        steps = [item.strip(" .") + "." for item in raw_items if item.strip()]

    return intro, steps


def render_guidance_card(source: str, text: str):
    """Renders one retrieved RAG chunk as a structured card instead of
    a single dense paragraph of prose."""
    intro, steps = _parse_guidance_text(text)

    intro_html = f'<div class="mv-guidance-intro">{intro}</div>' if intro else ""

    steps_html = ""
    if steps:
        step_rows = "".join(
            f"""<div class="mv-guidance-step">
                    <div class="mv-guidance-step-icon">{i+1:02d}</div>
                    <div>{step}</div>
                </div>"""
            for i, step in enumerate(steps)
        )
        steps_html = f'<div class="mv-guidance-steps-label">Suggested next steps</div>{step_rows}'

    render_html(
        f"""
        <div class="mv-guidance-card">
            <div class="mv-guidance-source">{source}</div>
            {intro_html}
            {steps_html}
        </div>
        """
    )


_REPORT_SECTION_ICONS = {
    "impression": "\U0001F3AF",
    "findings": "\U0001F52C",
    "suggested next steps": "\U0001F4CB",
    "next steps": "\U0001F4CB",
    "disclaimer": "\u26A0\ufe0f",
}


def _parse_report_sections(report_text: str):
    """Splits the LLM's drafted report into (heading, body) sections
    based on the section headers requested in the prompt template
    (Impression / Findings / Suggested Next Steps / Disclaimer), so
    each can be rendered as its own styled card instead of one long
    unstructured block of text. Falls back to a single 'Report' section
    if the model didn't follow the requested structure exactly."""
    pattern = re.compile(
        r"^#{0,3}\s*\**\s*(\d+[\.\)]\s*)?"
        r"(Impression|Findings|Suggested Next Steps|Next Steps|Disclaimer)"
        r"\s*\**:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(report_text))

    if not matches:
        return [("Report", report_text.strip())]

    sections = []
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report_text)
        body = report_text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _inline_markdown(text: str) -> str:
    """Converts basic inline markdown (bold/italic) to HTML."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"<em>\1</em>", text)
    return text


def _markdown_body_to_html(text: str) -> str:
    """Minimal markdown-to-HTML conversion for LLM-generated report
    bodies (paragraphs, '-'/'*' bullets, numbered lists, bold/italic).
    Needed because the whole report section -- heading, wrapper div,
    and body -- must be emitted as a single HTML string in one
    st.markdown() call (Streamlit does not let an opened tag in one
    call be closed by a later call; each call is an independent DOM
    fragment), so raw markdown syntax can't be handed to a separate
    st.markdown() call the way it normally would be."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    html_parts = []
    list_open = None  # "ul" | "ol" | None

    def close_list():
        nonlocal list_open
        if list_open:
            html_parts.append(f"</{list_open}>")
            list_open = None

    for line in lines:
        bullet = re.match(r"^[-*]\s+(.*)", line)
        numbered = re.match(r"^\d+[\.\)]\s+(.*)", line)
        if bullet:
            if list_open != "ul":
                close_list()
                html_parts.append("<ul>")
                list_open = "ul"
            html_parts.append(f"<li>{_inline_markdown(bullet.group(1))}</li>")
        elif numbered:
            if list_open != "ol":
                close_list()
                html_parts.append("<ol>")
                list_open = "ol"
            html_parts.append(f"<li>{_inline_markdown(numbered.group(1))}</li>")
        else:
            close_list()
            html_parts.append(f"<p>{_inline_markdown(line)}</p>")
    close_list()
    return "".join(html_parts)


def render_report(report_text: str):
    """Renders the LLM-drafted report as structured, icon-labeled
    section cards (Impression / Findings / Next Steps / Disclaimer)
    instead of one plain block of text, and provides a download button
    so the report can be saved as a .txt file."""
    sections = _parse_report_sections(report_text)

    for heading, body in sections:
        icon = _REPORT_SECTION_ICONS.get(heading.lower(), "\U0001F4C4")
        is_disclaimer = "disclaimer" in heading.lower()
        card_class = "mv-report-section disclaimer" if is_disclaimer else "mv-report-section"
        body_html = _markdown_body_to_html(body)

        render_html(
            f"""
            <div class="{card_class}">
                <div class="mv-report-section-title">{icon} {heading}</div>
                <div class="mv-report-section-body">{body_html}</div>
            </div>
            """
        )

    st.download_button(
        label="\u2B07 Download report (.txt)",
        data=report_text,
        file_name="medivision_ai_report.txt",
        mime="text/plain",
    )
