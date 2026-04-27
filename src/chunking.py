"""
Chunking strategies. Naive fixed-size chunking is the baseline; the interesting
work is showing why it breaks and what the alternatives fix.
"""

from typing import Iterator
from config import ChunkConfig


def chunk_fixed(text: str, cfg: ChunkConfig) -> list[str]:
    """Sliding-window fixed-size split. Baseline — fast but context-blind."""
    raise NotImplementedError


def chunk_recursive(text: str, cfg: ChunkConfig) -> list[str]:
    """Recursive character split on paragraph / sentence / word boundaries."""
    raise NotImplementedError


def chunk_semantic(text: str, cfg: ChunkConfig) -> list[str]:
    """
    Embed sentences, split at embedding-distance peaks.
    Addresses the main failure of fixed chunking: slicing mid-thought.
    """
    raise NotImplementedError


def chunk_document(doc: dict, cfg: ChunkConfig) -> Iterator[dict]:
    """
    Dispatch to the right strategy and yield chunk dicts:
    {"id": str, "text": str, "metadata": dict}
    """
    raise NotImplementedError
