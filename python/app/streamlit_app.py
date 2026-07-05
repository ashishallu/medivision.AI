"""
streamlit_app.py
MediVision AI - Streamlit interface.

Pipeline: upload chest X-ray -> ResNet-50 classifies findings ->
Grad-CAM highlights the driving region -> ChromaDB RAG retrieves
relevant clinical guidance -> Ollama (Mistral-7B) drafts a
doctor-readable report.

Visual theme lives in frontend/css/theme.css, injected via
python/utils/theme.py -- kept separate from this file so pipeline
logic and markup construction don't tangle together.

Run from the project root:
    streamlit run python/app/streamlit_app.py
"""

import sys
import os

# Allow `python.*` package imports when Streamlit runs this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import torch

from python.model.classifier import load_model, predict
from python.model.gradcam import GradCAM
from python.rag.retriever import retrieve_for_findings
from python.llm.generate_report import generate_report
from python.utils.preprocess import pil_from_uploaded_file, load_image_as_tensor
from python.utils.config_loader import load_config
from python.utils import theme

_cfg = load_config()

st.set_page_config(
    page_title=_cfg["streamlit"]["page_title"],
    page_icon=_cfg["streamlit"]["page_icon"],
    layout=_cfg["streamlit"]["layout"],
)

theme.inject_theme()


@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    return model, device


def _check_ollama_status() -> bool:
    """Quick, short-timeout check of whether Ollama is reachable, used
    only for the System Status HUD panel -- never blocks the pipeline."""
    import requests
    from python.llm.generate_report import _OLLAMA_HOST
    try:
        requests.get(_OLLAMA_HOST, timeout=1.5)
        return True
    except requests.exceptions.RequestException:
        return False


def main():
    theme.render_floating_nav([
        ("sec-input", "Input"),
        ("sec-explain", "Explain"),
        ("sec-detect", "Detect"),
        ("sec-retrieval", "Guidance"),
        ("sec-report", "Report"),
    ])

    theme.render_header(
        title="MEDIVISION.AI",
        subtitle="Analyze your chest X-rays in a click.",
        status_label="SYSTEM ONLINE",
    )

    theme.render_about_section()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ollama_up = _check_ollama_status()

    theme.render_hero_scan(
        model_info={
            "Backbone": "ResNet-50",
            "Classes": len(_cfg["model"]["class_names"]),
            "Input size": f"{_cfg['model']['input_size']}px",
        },
        dataset_info={
            "Source": "NIH ChestX-ray14",
            "Sample": "5,606 imgs",
            "Mean AUC": "0.72",
        },
        pipeline_steps=[
            "ResNet-50 classifier",
            "Grad-CAM explainability",
            "ChromaDB RAG retrieval",
            "Ollama report drafting",
        ],
        status_rows=[
            ("Compute", device.upper(), "ok" if device == "cuda" else "warn"),
            ("Ollama", "ONLINE" if ollama_up else "OFFLINE", "ok" if ollama_up else "warn"),
            ("Threshold", f"{_cfg['model']['prediction_threshold']:.2f}", ""),
        ],
    )

    with st.container(border=True):
        ctrl_col1, ctrl_col2 = st.columns([1, 3])
        with ctrl_col1:
            theme.control_label("CONFIDENCE THRESHOLD")
        with ctrl_col2:
            threshold = st.slider(
                "Prediction confidence threshold",
                min_value=0.1, max_value=0.9,
                value=float(_cfg["model"]["prediction_threshold"]), step=0.05,
                label_visibility="collapsed",
            )

    st.warning(
        "Research/portfolio demo only. Not a medical device. "
        "Every output requires physician review."
    )

    theme.render_anchor("sec-input")
    uploaded_file = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload an image to run the pipeline, or try a sample from data/sample_images/.")
        return

    image = pil_from_uploaded_file(uploaded_file)
    input_tensor = load_image_as_tensor(image)

    model, device = get_model()

    col1, col2 = st.columns(2)
    with col1:
        theme.render_eyebrow("INPUT // 01")
        st.subheader("Uploaded X-ray")
        theme.render_scan_frame(image, label="SOURCE IMAGE")

    with st.spinner("Running classifier..."):
        findings = predict(model, input_tensor, device=device)

    sorted_findings = dict(sorted(findings.items(), key=lambda x: x[1], reverse=True))
    positive = {k: v for k, v in sorted_findings.items() if v >= threshold}

    with col2:
        theme.render_anchor("sec-explain")
        theme.render_eyebrow("EXPLAINABILITY // 02")
        st.subheader("Grad-CAM Explainability")
        if positive:
            top_finding = next(iter(positive))
            class_names = _cfg["model"]["class_names"]
            class_index = class_names.index(top_finding)

            gradcam = GradCAM(model)
            overlay = gradcam.overlay_on_image(input_tensor, class_index)
            theme.render_scan_frame(overlay, label=f"FOCUS: {top_finding.upper()}")
        else:
            st.write("No findings above threshold to visualize.")

    theme.render_anchor("sec-detect")
    theme.render_eyebrow("CLASSIFIER OUTPUT // 03")
    st.subheader("Detection Confidence")
    theme.render_signal_meters(sorted_findings, threshold)

    if not positive:
        st.success("No findings crossed the confidence threshold.")
        return

    theme.render_anchor("sec-retrieval")
    theme.render_eyebrow("RETRIEVAL // 04")
    st.subheader("Retrieved Clinical Guidance (RAG)")
    try:
        guidance = retrieve_for_findings(findings, threshold=threshold)
        for finding, chunks in guidance.items():
            with st.expander(f"Guidance for: {finding}"):
                for chunk in chunks:
                    theme.render_guidance_card(chunk["source"], chunk["text"])
    except RuntimeError as exc:
        st.error(str(exc))
        guidance = {}

    theme.render_anchor("sec-report")
    theme.render_eyebrow("REPORT GENERATION // 05")
    st.subheader("AI-Drafted Diagnostic Report")
    if st.button("Generate report with Ollama (Mistral-7B)"):
        with st.spinner("Generating report locally via Ollama..."):
            try:
                report = generate_report(findings, guidance)
                theme.render_report(report)
            except RuntimeError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
