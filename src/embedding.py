"""
Embed chunks with a sentence-transformer model.
Handles batching and returns numpy arrays.
"""

import numpy as np
from config import EmbeddingConfig
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=None)
def load_model(cfg: EmbeddingConfig) -> SentenceTransformer:
    """Load and return the SentenceTransformer model."""
    return SentenceTransformer(
        cfg.model_name
    )

def embed_chunks(texts: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    """Return (N, D) embedding matrix for a list of text chunks."""
    model = load_model(
        cfg
    )

    embeddings = model.encode(
        texts, batch_size=cfg.batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    return embeddings
