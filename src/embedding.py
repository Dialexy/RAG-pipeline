"""
Embed chunks with a sentence-transformer model.
Handles batching and returns numpy arrays.
"""

import numpy as np
from ..config import EmbeddingConfig
from sentence_transformers import SentenceTransformer


def load_model(cfg: EmbeddingConfig) -> SentenceTransformer:
    """Load and return the SentenceTransformer model."""
    return SentenceTransformer(
        cfg.model_name
    )  # TODO: change to actual LLM name in config.py once ready to run


def embed_chunks(texts: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    """Return (N, D) embedding matrix for a list of text chunks."""
    model = load_model(
        cfg
    )  # TODO: Currently inefficient, come back later if processing slows

    embeddings = model.encode(
        texts, batch_size=cfg.batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    return embeddings
