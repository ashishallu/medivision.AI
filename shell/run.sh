#!/usr/bin/env bash
# run.sh
# Launches the MediVision AI Streamlit app.
# Run from the project root: bash shell/run.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

echo "[run] Launching Streamlit app..."
streamlit run python/app/streamlit_app.py
