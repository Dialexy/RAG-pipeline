"""
Retrieval layer: dense search, BM25, hybrid fusion, and cross-encoder re-ranking.

This is where naive RAG breaks down:
  - dense-only misses exact-match queries  →  add BM25 (hybrid)
  - top-k by similarity ≠ top-k by relevance  →  add cross-encoder re-ranker
  - context window pressure  →  pick rerank_top_n << top_k
"""

from config import RetrievalConfig


def bm25_search(query: str, corpus: list[dict], top_k: int) -> list[dict]:
    """Sparse BM25 retrieval over the full chunk corpus."""
    raise NotImplementedError


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists via RRF before re-ranking."""
    raise NotImplementedError


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Cross-encoder re-ranking — returns top_n chunks sorted by relevance."""
    raise NotImplementedError


def retrieve(
    query: str, collection, corpus: list[dict], cfg: RetrievalConfig
) -> list[dict]:
    """
    Full retrieval pipeline for a single query:
    dense → (BM25 → RRF) → re-rank → return top-n chunks
    """
    raise NotImplementedError
