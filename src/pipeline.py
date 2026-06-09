"""
End-to-end pipeline: index-time and query-time entry points.
"""

import time
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from config import (
    CHROMA_PERSIST_DIR,
    PipelineConfig,
    RAW_DIR,
)
from .models import Document
from .ingestion import fetch_corpus, iter_documents
from .vector_store import build_index, load_index, load_chunk_corpus
from .chunking import chunk_document
from .retrieval import retrieve
from .generation import generate, ABSTENTION_RESPONSE
from .logger import get_logger

logger = get_logger(__name__)


def file_hash(path: Path) -> str:
    hash = hashlib.sha256()
    hash.update(path.name.encode())
    hash.update(b"\0")
    hash.update(path.read_bytes())
    return hash.hexdigest()


def corpus_hash(raw_dir: Path) -> str:
    file_path = sorted(list(raw_dir.rglob("*.txt")) + list(raw_dir.rglob("*.md")))

    hash = hashlib.sha256()

    for file in file_path:
        hashed_file = file_hash(file)
        hash.update(hashed_file.encode())
        hash.update(b"\0")

    return hash.hexdigest()


def config_fingerprint(cfg: PipelineConfig) -> str:
    hash = hashlib.sha256()
    configs_as_dicts = {
        "ChunkingConfig": asdict(cfg.chunking),
        "EmbeddingConfig": asdict(cfg.embedding),
    }

    configs_as_json = json.dumps(configs_as_dicts, sort_keys=True)

    hash.update(configs_as_json.encode())
    return hash.hexdigest()


def build_pipeline(cfg: PipelineConfig, force=False) -> None:
    """
    Index-time: ingest → chunk → embed → store.
    Run once (or when the corpus changes).
    """

    t_start = time.perf_counter()
    logger.info("Starting indexing pipeline")

    try:
        fetch_corpus(RAW_DIR)
    except Exception as e:
        logger.error("Corpus fetch failed: %s", e)
        raise RuntimeError("Corpus fetch failed - check network connection") from e

    current_hash = corpus_hash(RAW_DIR) + config_fingerprint(cfg)
    hash_file = CHROMA_PERSIST_DIR / "corpus.hash"

    if not force and hash_file.exists():
        stored_hash = hash_file.read_text()

        if current_hash == stored_hash:
            logger.info("Indexing Skipped, corpus unchanged")
            return

    documents = list(iter_documents(RAW_DIR))
    if not documents:
        raise RuntimeError(f"No documents found in {RAW_DIR}")
    logger.info("Ingested %d documents", len(documents))

    t_chunk = time.perf_counter()
    chunks = []
    total_docs = len(documents)
    for i, document in enumerate(documents, start=1):
        for chunk in chunk_document(document, cfg.chunking, cfg.embedding):
            chunks.append(chunk)
        if i % 1000 == 0:
            logger.info("Chunked %d / %d documents", i, total_docs)
    logger.info(
        "Chunking complete: %d chunks in %.1fs",
        len(chunks),
        time.perf_counter() - t_chunk,
    )

    t_index = time.perf_counter()
    try:
        build_index(chunks, cfg)
    except Exception as e:
        logger.error("Index build failed: %s", e)
        raise RuntimeError("Index build failed - ChromaDB may be corrupted") from e
    logger.info("Index built in %.1fs", time.perf_counter() - t_index)

    logger.info("Indexing pipeline complete in %.1fs", time.perf_counter() - t_start)
    hash_file.write_text(current_hash)


def query_pipeline(
    query: str,
    cfg: PipelineConfig,
    collection=None,
    corpus: list[Document] | None = None,
    filters: dict | None = None,
) -> dict:
    """
    Query-time: embed query → retrieve → generate.
    Returns {"query": str, "answer": str, "sources": list[dict], "error": str | None}
    """
    t_start = time.perf_counter()
    logger.info("Query received: %r", query)

    if collection is None:
        collection = load_index(cfg)

    if corpus is None:
        corpus = load_chunk_corpus(collection)

    t_retrieve = time.perf_counter()
    try:
        chunks = retrieve(query, collection, corpus, cfg, filters=filters)
        logger.info(
            "Retrieval took %.2fs, %d chunks",
            time.perf_counter() - t_retrieve,
            len(chunks),
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {
            "query": query,
            "answer": "I encountered an error during retrieval. Please try again.",
            "sources": [],
            "error": str(e),
        }

    if (
        cfg.retrieval.use_reranker
        and chunks
        and chunks[0].get("reranked_score") is not None
        and chunks[0]["reranked_score"] < cfg.retrieval.reranker_score_threshold
    ):
        logger.info(
            "Top reranker score %.3f below threshold %.3f, returning abstention",
            chunks[0]["reranked_score"],
            cfg.retrieval.reranker_score_threshold,
        )
        return {
            "query": query,
            "answer": ABSTENTION_RESPONSE,
            "sources": [],
            "error": None,
        }

    t_generate = time.perf_counter()
    try:
        answer = generate(query, chunks, cfg.generation)
        logger.info("Generation took %.2fs", time.perf_counter() - t_generate)
    except RuntimeError as e:
        logger.error("Generation failed: %s", e)
        return {
            "query": query,
            "answer": str(e),
            "sources": chunks,
            "error": str(e),
        }

    logger.info("Query complete in %.2fs", time.perf_counter() - t_start)

    return {
        "query": query,
        "answer": answer,
        "sources": chunks,
        "error": None,
    }
