"""
Embed chunks with a sentence-transformer model.
Handles batching and returns numpy arrays.
"""

import numpy as np
from config import EmbeddingConfig
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from .logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=None)
def load_model(cfg: EmbeddingConfig) -> SentenceTransformer:
    """Load and return the SentenceTransformer model."""
    logger.info("Loading embedding model: %s", cfg.model_name)
    return SentenceTransformer(cfg.model_name)


def embed_chunks(texts: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    """Return (N, D) embedding matrix for a list of text chunks."""
    model = load_model(cfg)

    logger.info("Embedding %d texts (batch_size=%d)", len(texts), cfg.batch_size)
    embeddings = model.encode(
        texts, batch_size=cfg.batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    logger.info("Embedding complete: shape %s", embeddings.shape)

    return embeddings
