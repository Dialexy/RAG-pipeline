"""
Chroma wrapper: index chunks and run similarity search.
"""

from config import PipelineConfig


def build_index(chunks: list[dict], cfg: PipelineConfig) -> None:
    """
    Embed all chunks and upsert into Chroma.
    Persists to CHROMA_PERSIST_DIR.
    """
    raise NotImplementedError


def load_index(cfg: PipelineConfig):
    """Return an existing Chroma collection from disk."""
    raise NotImplementedError


def dense_search(query_embedding, top_k: int, collection) -> list[dict]:
    """Return top-k chunks by cosine similarity."""
    raise NotImplementedError
