"""
Retrieval layer: dense search, BM25, hybrid fusion, and cross-encoder re-ranking.

This is where naive RAG breaks down:
  - dense-only misses exact-match queries  →  add BM25 (hybrid)
  - top-k by similarity ≠ top-k by relevance  →  add cross-encoder re-ranker
  - context window pressure  →  pick rerank_top_n << top_k
"""

from config import PipelineConfig, GenerationConfig
from .models import Document
from .embedding import embed_chunks
from .vector_store import dense_search, fetch_neighbouring_chunks
from rank_bm25 import BM25Okapi
from collections import defaultdict
from sentence_transformers import CrossEncoder
from functools import lru_cache
from .logger import get_logger
import numpy as np
import ollama
import httpx

logger = get_logger(__name__)


@lru_cache(maxsize=None)
def load_rank() -> CrossEncoder:
    try:
        model = CrossEncoder("BAAI/bge-reranker-large")
        # Force a small forward pass to confirm GPU allocation succeeds
        model.predict([("test", "test")])
        return model
    except Exception as e:
        if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
            logger.warning("Reranker GPU load failed (%s), falling back to CPU", e)
            return CrossEncoder("BAAI/bge-reranker-large", device="cpu")
        raise


@lru_cache(maxsize=None)
def build_bm25_index(corpus: tuple[Document, ...]) -> BM25Okapi:
    tokenised_corpus = [doc.text.lower().split() for doc in corpus]
    return BM25Okapi(tokenised_corpus)


def bm25_search(query: str, corpus: list[Document], top_k: int) -> list[dict]: #TODO: Change the granuality to chunks not files. Right now dense serach is more granualar.
    """Sparse BM25 retrieval over the full chunk corpus."""

    bm25 = build_bm25_index(tuple(corpus))

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "text": corpus[i].text,
            "metadata": corpus[i].metadata,
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = defaultdict(float)
    doc_map = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_map[doc["text"]] = doc
            scores[doc["text"]] += 1 / (k + rank)

    sorted_items = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [{**doc_map[text], "score": score} for text, score in sorted_items]


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Cross-encoder re-ranking; returns top_n chunks sorted by relevance."""
    model = load_rank()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            **candidate,
            "reranked_score": float(score),
        }
        for candidate, score in ranked[:top_n]
    ]


def expand_query(query: str, cfg: GenerationConfig) -> list[str]:
    """Ask the LLM for 2 alternative phrasings; returns original + up to 2 variants."""
    prompt = f"""Generate 2 alternative phrasings of this question that mean the same thing.
Return only the questions, one per line, no numbering or explanation.

Question: {query}"""

    try:
        response = ollama.chat(
            model=cfg.model, messages=[{"role": "user", "content": prompt}]
        )
        raw = response.message.content or ""
        variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return [query] + variants[:2]
    except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
        logger.warning("Query expansion failed, continuing with original query: %s", e)
        return [query]


def retrieve(
    query: str, collection, corpus: list[Document], cfg: PipelineConfig, filters: dict | None = None
) -> list[dict]:
    """
    Full retrieval pipeline for a single query:
    expand → dense → (BM25) → RRF → re-rank → return top-n chunks
    """

    logger.info(
        "Retrieving for query: %r (hybrid=%s, reranker=%s, query_expansion=%s, top_k=%d)",
        query,
        cfg.retrieval.use_hybrid,
        cfg.retrieval.use_reranker,
        cfg.retrieval.use_query_expansion,
        cfg.retrieval.top_k,
    )

    if cfg.retrieval.use_query_expansion:
        query_variants = expand_query(query, cfg.generation)
        logger.info(
            "Query expanded to %d variants: %s", len(query_variants), query_variants
        )
    else:
        query_variants = [query]

    all_dense = []
    all_bm25 = []

    active_filters = filters if filters is not None else cfg.retrieval.default_filters

    for variant in query_variants:
        embedding = embed_chunks([variant], cfg.embedding)[0]
        all_dense.append(dense_search(embedding, cfg.retrieval.top_k, collection, filters=active_filters))

        if cfg.retrieval.use_hybrid:
            all_bm25.append(bm25_search(variant, corpus, cfg.retrieval.top_k))

    if cfg.retrieval.use_hybrid:
        candidates = reciprocal_rank_fusion(all_dense + all_bm25)
    else:
        candidates = reciprocal_rank_fusion(all_dense)

    if cfg.retrieval.use_reranker:
        results = rerank(query, candidates, cfg.retrieval.rerank_top_n)
    else:
        results = candidates[: cfg.retrieval.rerank_top_n]

    for result in results:
        meta = result["metadata"]
        if "parent_id" in meta and "chunk_index" in meta:
            chunk_id = f"{meta['parent_id']}::chunk{meta['chunk_index']}"
            neighbours = fetch_neighbouring_chunks(chunk_id, collection)
            result["text"] = "\n".join(neighbours[:1] + [result["text"]] + neighbours[1:])

    logger.info("Retrieval complete: %d chunks returned", len(results))
    return results
