"""
streamlit_app.py
MediVision AI - Streamlit interface.

Pipeline: upload chest X-ray -> ResNet-50 classifies findings ->
Grad-CAM highlights the driving region -> ChromaDB RAG retrieves
relevant clinical guidance -> Ollama (Mistral-7B) drafts a
doctor-readable report.

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

_cfg = load_config()

st.set_page_config(
    page_title=_cfg["streamlit"]["page_title"],
    page_icon=_cfg["streamlit"]["page_icon"],
    layout=_cfg["streamlit"]["layout"],
)


@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    return model, device


def main():
    st.title("🫁 MediVision AI")
    st.caption("CNN classifier + Grad-CAM explainability + RAG-grounded diagnostic drafting")

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Prediction confidence threshold",
            min_value=0.1, max_value=0.9,
            value=float(_cfg["model"]["prediction_threshold"]), step=0.05,
        )
        st.divider()
        st.markdown(
            "**Pipeline**\n"
            "1. ResNet-50 multi-label classifier\n"
            "2. Grad-CAM visual explainability\n"
            "3. ChromaDB + sentence-transformers RAG\n"
            "4. Ollama (Mistral-7B) report drafting\n"
        )
        st.warning(
            "Research/portfolio demo only. Not a medical device. "
            "Every output requires physician review."
        )

    uploaded_file = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload an image to run the pipeline, or try a sample from data/sample_images/.")
        return

    image = pil_from_uploaded_file(uploaded_file)
    input_tensor = load_image_as_tensor(image)

    model, device = get_model()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    with st.spinner("Running classifier..."):
        findings = predict(model, input_tensor, device=device)

    sorted_findings = dict(sorted(findings.items(), key=lambda x: x[1], reverse=True))
    positive = {k: v for k, v in sorted_findings.items() if v >= threshold}

    with col2:
        st.subheader("Grad-CAM Explainability")
        if positive:
            top_finding = next(iter(positive))
            class_names = _cfg["model"]["class_names"]
            class_index = class_names.index(top_finding)

            gradcam = GradCAM(model)
            overlay = gradcam.overlay_on_image(input_tensor, class_index)
            st.image(overlay, caption=f"Highlighted region driving: {top_finding}", use_container_width=True)
        else:
            st.write("No findings above threshold to visualize.")

    st.subheader("Classifier Output")
    st.bar_chart(sorted_findings)

    if not positive:
        st.success("No findings crossed the confidence threshold.")
        return

    st.subheader("Retrieved Clinical Guidance (RAG)")
    try:
        guidance = retrieve_for_findings(findings, threshold=threshold)
        for finding, chunks in guidance.items():
            with st.expander(f"Guidance for: {finding}"):
                for chunk in chunks:
                    st.markdown(f"**Source: {chunk['source']}**")
                    st.write(chunk["text"])
    except RuntimeError as exc:
        st.error(str(exc))
        guidance = {}

    st.subheader("AI-Drafted Diagnostic Report")
    if st.button("Generate report with Ollama (Mistral-7B)"):
        with st.spinner("Generating report locally via Ollama..."):
            try:
                report = generate_report(findings, guidance)
                st.markdown(report)
            except RuntimeError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
