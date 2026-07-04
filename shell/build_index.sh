#!/usr/bin/env bash
# build_index.sh
# Builds/rebuilds the ChromaDB RAG index from data/clinical_guidelines/.
# Run from the project root: bash shell/build_index.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

echo "[build_index] Building RAG index..."
python -m python.rag.build_index
