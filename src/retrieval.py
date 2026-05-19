"""
Retrieval layer: dense search, BM25, hybrid fusion, and cross-encoder re-ranking.

This is where naive RAG breaks down:
  - dense-only misses exact-match queries  →  add BM25 (hybrid)
  - top-k by similarity ≠ top-k by relevance  →  add cross-encoder re-ranker
  - context window pressure  →  pick rerank_top_n << top_k
"""

from config import PipelineConfig
from .models import Document
from .embedding import embed_chunks
from .vector_store import dense_search
from rank_bm25 import BM25Okapi
from collections import defaultdict
from sentence_transformers import CrossEncoder
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=None)
def load_rank() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def bm25_search(query: str, corpus: list[Document], top_k: int) -> list[dict]:
    """Sparse BM25 retrieval over the full chunk corpus."""

    tokenised_corpus = [doc.text.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised_corpus)

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
    """Cross-encoder re-ranking — returns top_n chunks sorted by relevance."""
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


def retrieve(
    query: str, collection, corpus: list[Document], cfg: PipelineConfig
) -> list[dict]:
    """
    Full retrieval pipeline for a single query:
    dense → (BM25 → RRF) → re-rank → return top-n chunks
    """

    embedding_matrix = embed_chunks([query], cfg.embedding)
    query_embedding = embedding_matrix[0]

    dense_result = dense_search(query_embedding, cfg.retrieval.top_k, collection)

    if cfg.retrieval.use_hybrid:
        bm25_result = bm25_search(query, corpus, cfg.retrieval.top_k)
        candidates = reciprocal_rank_fusion([dense_result, bm25_result])

    else:
        candidates = dense_result

    if cfg.retrieval.use_reranker:
        return rerank(query, candidates, cfg.retrieval.rerank_top_n)
    else:
        return candidates[: cfg.retrieval.rerank_top_n]
