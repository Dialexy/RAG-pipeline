"""
Embed chunks with a sentence-transformer model.
Handles batching and returns numpy arrays.
"""

import numpy as np
from config import EmbeddingConfig


def load_model(cfg: EmbeddingConfig):
    """Load and return the SentenceTransformer model."""
    raise NotImplementedError


def embed_chunks(texts: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    """Return (N, D) embedding matrix for a list of text chunks."""
    raise NotImplementedError
