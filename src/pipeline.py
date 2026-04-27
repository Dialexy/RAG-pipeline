"""
End-to-end pipeline: index-time and query-time entry points.
"""

from config import PipelineConfig


def build_pipeline(cfg: PipelineConfig) -> None:
    """
    Index-time: ingest → chunk → embed → store.
    Run once (or when the corpus changes).
    """
    raise NotImplementedError


def query_pipeline(query: str, cfg: PipelineConfig) -> dict:
    """
    Query-time: embed query → retrieve → generate.
    Returns {"query": str, "answer": str, "sources": list[dict]}
    """
    raise NotImplementedError
