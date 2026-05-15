"""
End-to-end pipeline: index-time and query-time entry points.
"""

from config import PipelineConfig, PROCESSED_DIR, RAW_DIR
from .ingestion import fetch_corpus, iter_documents
from .chunking import chunk_document
from .vector_store import build_index, load_index
from .retrieval import retrieve
from .generation import generate

def build_pipeline(cfg: PipelineConfig) -> None:
    """
    Index-time: ingest → chunk → embed → store.
    Run once (or when the corpus changes).
    """
    fetch_corpus(RAW_DIR)
    documents = list(iter_documents(RAW_DIR))
    chunks = []

    for document in documents:
        for chunk in chunk_document(document, cfg.chunking):
            chunks.append(chunk)

    build_index(chunks, cfg)

def query_pipeline(query: str, cfg: PipelineConfig) -> dict:
    """
    Query-time: embed query → retrieve → generate.
    Returns {"query": str, "answer": str, "sources": list[dict]}
    """
    collection = load_index(cfg)

    corpus = list(iter_documents(RAW_DIR))
    chunks = retrieve(query, collection, corpus, cfg)
    answer = generate(query, chunks, cfg.generation)

    return {
            "query": query,
            "answer": answer,
            "sources": chunks,
            }
