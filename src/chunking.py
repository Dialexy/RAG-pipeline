"""
Chunking strategies. Naive fixed-size chunking is the baseline; the interesting
work is showing why it breaks and what the alternatives fix.
"""

from collections.abc import Iterator
from config import ChunkConfig
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


def chunk_recursive(text: str, cfg: ChunkConfig, _separators: list[str] | None = None) -> list[str]:
    """Recursive character split on paragraph / sentence / word boundaries."""

    if len(text) <= cfg.chunk_size:
        return [text]

    if _separators is None:
        _separators = ["\n\n", "\n", ". ", " "]

    results = []

    for i, seperator in enumerate(_separators):
        splittext = text.split(seperator)

        for part in splittext:
            if not part.strip():
                continue
            if len(part) <= cfg.chunk_size:
                results.append(part)
            else:
                remaining = _separators[i + 1:]
                if remaining and any(sep in part for sep in remaining):
                    sub_chunk = chunk_recursive(part, cfg, remaining)
                else:
                    step = max(1, cfg.chunk_size - cfg.chunk_overlap)
                    sub_chunk = [part[i : i + cfg.chunk_size] for i in range(0, len(part), step)]
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

    if cfg.strategy == "fixed":
        texts = chunk_fixed(doc.text, cfg)
    elif cfg.strategy == "recursive":
        texts = chunk_recursive(doc.text, cfg)
    else:
        raise ValueError(f"Unknown: {cfg.strategy}")

    for i, chunk_text in enumerate(texts):
        yield Document(
            id=f"{doc.id}::chunk{i}",
            text=chunk_text,
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "chunk_strategy": cfg.strategy,
                "parent_id": doc.id,
            },
        )
