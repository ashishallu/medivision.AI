"""
build_index.py
Reads clinical guideline text files from data/clinical_guidelines/,
chunks them, embeds them with a sentence-transformers model, and
stores them in a persistent ChromaDB collection.

Run once (and again any time guideline documents change):
    python -m python.rag.build_index
"""

import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer

from python.utils.config_loader import load_config, resolve_path

_cfg = load_config()


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Simple sliding-window chunker over words (fast, dependency-free,
    good enough for short guideline documents)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end - overlap > start else end
    return chunks


def build_index():
    guidelines_dir = resolve_path(_cfg["paths"]["clinical_guidelines_dir"])
    persist_dir = resolve_path(_cfg["paths"]["chroma_persist_dir"])
    os.makedirs(persist_dir, exist_ok=True)

    files = glob.glob(os.path.join(guidelines_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt guideline files found in {guidelines_dir}. "
            "Add clinical reference documents there before building the index."
        )

    print(f"[build_index] Loading embedding model: {_cfg['rag']['embedding_model']}")
    embedder = SentenceTransformer(_cfg["rag"]["embedding_model"])

    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = _cfg["rag"]["chroma_collection_name"]

    # Fresh build each run so stale chunks never linger
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    doc_id = 0
    for filepath in files:
        source_name = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, _cfg["rag"]["chunk_size"], _cfg["rag"]["chunk_overlap"])
        if not chunks:
            continue

        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{source_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        doc_id += len(chunks)
        print(f"[build_index] Indexed {len(chunks)} chunks from {source_name}")

    print(f"[build_index] Done. Total chunks indexed: {doc_id}. Stored at {persist_dir}")


if __name__ == "__main__":
    build_index()
