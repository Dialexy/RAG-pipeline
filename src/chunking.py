"""
Chunking strategies. Naive fixed-size chunking is the baseline; the interesting
work is showing why it breaks and what the alternatives fix.
"""

from collections.abc import Iterator
from ..config import ChunkConfig
from .models import Document


def chunk_fixed(text: str, cfg: ChunkConfig) -> list[str]:
    """Sliding-window fixed-size split. Baseline — fast but context-blind."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + cfg.chunk_size
        chunks.append(text[start:end])
        start += cfg.chunk_size - cfg.chunk_overlap

    return chunks


def chunk_recursive(text: str, cfg: ChunkConfig) -> list[str]:
    """Recursive character split on paragraph / sentence / word boundaries."""
    seperators = ["\n\n", "\n", ". ", " ", ""]
    results = []

    for seperator in seperators:
        splittext = text.split(seperator)

        for part in splittext:
            if len(part) <= cfg.chunk_size:
                results.append(part)
            else:
                sub_chunk = chunk_recursive(part, cfg)
                results.extend(sub_chunk)

        if len(splittext) > 1:
            break

    merged_chunks = []
    for result in results:
        if len(merged_chunks) == 0:
            merged_chunks.append(result)
        else:
            previous_chunk = merged_chunks[-1]
            overlap = previous_chunk[-cfg.chunk_overlap :]
            combined = overlap + result
            merged_chunks.append(combined)

    return merged_chunks


def chunk_semantic(text: str, cfg: ChunkConfig) -> list[str]:
    """
    Embed sentences, split at embedding-distance peaks.
    Addresses the main failure of fixed chunking: slicing mid-thought.
    """
    raise NotImplementedError


def chunk_document(doc: Document, cfg: ChunkConfig) -> Iterator[Document]:
    """
    Dispatch to the right strategy and yield chunk dicts:
    {"id": str, "text": str, "metadata": dict}
    """
    raise NotImplementedError
