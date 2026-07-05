"""
retriever.py
Given a finding (e.g. "Cardiomegaly"), retrieves the most relevant
clinical guideline chunks from the persistent ChromaDB collection built
by build_index.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from python.utils.config_loader import load_config, resolve_path

_cfg = load_config()

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_cfg["rag"]["embedding_model"])
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        persist_dir = resolve_path(_cfg["paths"]["chroma_persist_dir"])
        client = chromadb.PersistentClient(path=persist_dir)
        collection_name = _cfg["rag"]["chroma_collection_name"]
        try:
            _collection = client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist yet -- most likely a fresh clone or
            # a fresh deployment (e.g. Streamlit Community Cloud) where
            # data/chroma_db/ is gitignored and was never built. Build it
            # automatically instead of requiring a manual CLI step that
            # isn't possible on most hosted deployments.
            from python.rag.build_index import build_index
            build_index()
            _collection = client.get_collection(collection_name)
    return _collection


def retrieve_guidance(query: str, top_k: int = None):
    """Returns a list of {text, source} dicts for the most relevant
    guideline chunks matching the query string (e.g. a finding name or
    a natural-language question)."""
    top_k = top_k or _cfg["rag"]["top_k_retrieval"]
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(documents, metadatas):
        hits.append({"text": doc, "source": meta.get("source", "unknown")})
    return hits


def retrieve_for_findings(findings: dict, threshold: float = None):
    """Convenience wrapper: given the classifier's {class_name: prob}
    output, retrieves guidance only for findings above the confidence
    threshold. Returns {finding_name: [chunks]}."""
    threshold = threshold if threshold is not None else _cfg["model"]["prediction_threshold"]
    guidance = {}
    for finding, prob in findings.items():
        if prob >= threshold:
            guidance[finding] = retrieve_guidance(finding)
    return guidance
