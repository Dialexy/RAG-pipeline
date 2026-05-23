"""
Chunking strategies. Naive fixed-size chunking is the baseline; the interesting
work is showing why it breaks and what the alternatives fix.
"""

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from .embedding import embed_chunks
from collections.abc import Iterator
from config import EmbeddingConfig
from config import ChunkConfig
from .models import Document
from .logger import get_logger

logger = get_logger(__name__)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


def chunk_fixed(text: str, cfg: ChunkConfig) -> list[str]:
    """Sliding-window fixed-size split. Baseline, fast but context-blind."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + cfg.chunk_size
        chunks.append(text[start:end])
        start += cfg.chunk_size - cfg.chunk_overlap

    return chunks


def chunk_recursive(
    text: str, cfg: ChunkConfig, _separators: list[str] | None = None
) -> list[str]:
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
                remaining = _separators[i + 1 :]
                if remaining and any(sep in part for sep in remaining):
                    sub_chunk = chunk_recursive(part, cfg, remaining)
                else:
                    step = max(1, cfg.chunk_size - cfg.chunk_overlap)
                    sub_chunk = [
                        part[i : i + cfg.chunk_size] for i in range(0, len(part), step)
                    ]
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
    sentences = sent_tokenize(text)

    if len(sentences) <= 1:
        return [text]

    embeddings = embed_chunks(sentences, EmbeddingConfig())

    distances = [
        1
        - (
            np.dot(embeddings[i], embeddings[i + 1])
            / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
        )
        for i in range(len(embeddings) - 1)
    ]

    threshold = float(np.mean(distances) + cfg.semantic_split_std_multiplier * np.std(distances))

    split_points = {i for i, d in enumerate(distances) if d > threshold}

    chunks: list[str] = []
    group: list[str] = []
    for i, sentence in enumerate(sentences):
        group.append(sentence)
        if i in split_points:
            chunks.append(" ".join(group))
            group = []
    if group:
        chunks.append(" ".join(group))

    result: list[str] = []
    for chunk in chunks:
        if len(chunk) > cfg.chunk_size:
            result.extend(chunk_recursive(chunk, cfg))
        else:
            result.append(chunk)

    return [chunk for chunk in result if len(chunk) >= cfg.semantic_min_chunk_size]


def chunk_document(doc: Document, cfg: ChunkConfig) -> Iterator[Document]:
    """
    Dispatch to the right strategy and yield chunk dicts:
    {"id": str, "text": str, "metadata": dict}
    """

    if cfg.strategy == "fixed":
        texts = chunk_fixed(doc.text, cfg)
    elif cfg.strategy == "recursive":
        texts = chunk_recursive(doc.text, cfg)
    elif cfg.strategy == "semantic":
        texts = chunk_semantic(doc.text, cfg)
    else:
        raise ValueError(f"Unknown: {cfg.strategy}")

    logger.debug(
        "Document %s → %d chunks (strategy=%s)", doc.id, len(texts), cfg.strategy
    )

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
