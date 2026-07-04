#!/usr/bin/env bash
# setup.sh
# One-time environment setup for MediVision AI.
# Run from the project root: bash shell/setup.sh

set -e  # exit immediately on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== MediVision AI setup ==="
echo "Project root: $PROJECT_ROOT"

# 1. Python virtual environment
if [ ! -d "venv" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[setup] Virtual environment already exists, skipping creation."
fi

echo "[setup] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "[setup] Upgrading pip..."
pip install --upgrade pip

echo "[setup] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. Ollama (free, local LLM runtime)
if ! command -v ollama &> /dev/null; then
    echo "[setup] Ollama not found."
    echo "[setup] Install it manually from https://ollama.com/download for your OS,"
    echo "[setup] then re-run this script to pull the model."
else
    echo "[setup] Ollama found. Pulling mistral:7b (this may take a while, ~4GB)..."
    ollama pull mistral:7b
fi

# 3. Create data directories that are gitignored / not shipped
mkdir -p data/chroma_db data/gradcam_outputs python/model/weights

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Add clinical guideline .txt files to data/clinical_guidelines/"
echo "  2. Build the RAG index:   python -m python.rag.build_index"
echo "  3. (Optional) Fine-tune the classifier: python -m python.model.train"
echo "  4. Start Ollama in another terminal:    ollama serve"
echo "  5. Run the app:                          bash shell/run.sh"
