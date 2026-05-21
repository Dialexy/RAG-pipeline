"""
End-to-end pipeline: index-time and query-time entry points.
"""

import time
from config import PipelineConfig, RAW_DIR
from .models import Document
from .ingestion import fetch_corpus, iter_documents
from .chunking import chunk_document
from .vector_store import build_index, load_index
from .retrieval import retrieve
from .generation import generate
from .logger import get_logger

logger = get_logger(__name__)


def build_pipeline(cfg: PipelineConfig) -> None:
    """
    Index-time: ingest → chunk → embed → store.
    Run once (or when the corpus changes).
    """
    t_start = time.perf_counter()
    logger.info("Starting indexing pipeline")

    fetch_corpus(RAW_DIR)
    documents = list(iter_documents(RAW_DIR))
    logger.info("Ingested %d documents", len(documents))

    t_chunk = time.perf_counter()
    chunks = []
    for document in documents:
        for chunk in chunk_document(document, cfg.chunking):
            chunks.append(chunk)
    logger.info(
        "Chunking complete: %d chunks in %.1fs",
        len(chunks),
        time.perf_counter() - t_chunk,
    )

    t_index = time.perf_counter()
    build_index(chunks, cfg)
    logger.info("Index built in %.1fs", time.perf_counter() - t_index)

    logger.info("Indexing pipeline complete in %.1fs", time.perf_counter() - t_start)


def query_pipeline(
    query: str,
    cfg: PipelineConfig,
    collection=None,
    corpus: list[Document] | None = None,
    filters: dict | None = None,
) -> dict:
    """
    Query-time: embed query → retrieve → generate.
    Returns {"query": str, "answer": str, "sources": list[dict]}
    """
    t_start = time.perf_counter()
    logger.info("Query received: %r", query)

    if collection is None:
        collection = load_index(cfg)

    if corpus is None:
        corpus = list(iter_documents(RAW_DIR))

    t_retrieve = time.perf_counter()
    chunks = retrieve(query, collection, corpus, cfg, filters=filters)
    logger.info("Retrieval took %.2fs, %d chunks", time.perf_counter() - t_retrieve, len(chunks))

    t_generate = time.perf_counter()
    answer = generate(query, chunks, cfg.generation)
    logger.info("Generation took %.2fs", time.perf_counter() - t_generate)

    logger.info("Query complete in %.2fs", time.perf_counter() - t_start)

    return {
        "query": query,
        "answer": answer,
        "sources": chunks,
    }
