"""
generate_report.py
Calls a locally-running Ollama server (free, no API key, runs on your
own machine) to turn classifier findings + retrieved clinical guidance
into a doctor-readable diagnostic report.

Prerequisite: Ollama installed and the model pulled, e.g.:
    ollama pull mistral:7b
    ollama serve
"""

import os
import requests

from python.utils.config_loader import load_config

_cfg = load_config()

# Allow docker-compose (or any deployment) to override the configured
# Ollama host via an environment variable, without editing config.yaml.
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", _cfg["llm"]["ollama_host"])


def _build_prompt(findings: dict, guidance: dict, threshold: float) -> str:
    positive_findings = {k: v for k, v in findings.items() if v >= threshold}

    if not positive_findings:
        findings_block = "No findings crossed the confidence threshold; the scan appears largely unremarkable."
    else:
        lines = [f"- {name}: {prob:.1%} model confidence" for name, prob in positive_findings.items()]
        findings_block = "\n".join(lines)

    guidance_lines = []
    for finding, chunks in guidance.items():
        for chunk in chunks:
            guidance_lines.append(f"[{finding} | source: {chunk['source']}] {chunk['text']}")
    guidance_block = "\n".join(guidance_lines) if guidance_lines else "No additional reference material retrieved."

    prompt = f"""You are assisting a radiologist by drafting a preliminary chest X-ray report.
Use ONLY the information below. Do not invent findings that are not listed.
Be clear that this is an AI-assisted draft requiring physician review, not a final diagnosis.

MODEL FINDINGS (with confidence scores):
{findings_block}

RETRIEVED CLINICAL REFERENCE MATERIAL:
{guidance_block}

Write a concise, structured draft report with these sections:
1. Impression (1-2 sentences)
2. Findings (bullet points, referencing confidence where relevant)
3. Suggested Next Steps (based on the reference material)
4. Disclaimer (state this is AI-generated and requires physician confirmation)
"""
    return prompt


def generate_report(findings: dict, guidance: dict) -> str:
    """Sends the prompt to the local Ollama server and returns the
    generated report text. Raises a clear error if Ollama isn't running,
    rather than failing silently."""
    threshold = _cfg["model"]["prediction_threshold"]
    prompt = _build_prompt(findings, guidance, threshold)

    url = f"{_OLLAMA_HOST}/api/generate"
    payload = {
        "model": _cfg["llm"]["model_name"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": _cfg["llm"]["temperature"],
            "num_predict": _cfg["llm"]["max_tokens"],
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{_OLLAMA_HOST}. Make sure Ollama is installed and "
            "running (`ollama serve`) and that the model has been pulled "
            f"(`ollama pull {_cfg['llm']['model_name']}`)."
        ) from exc

    data = response.json()
    return data.get("response", "").strip()
