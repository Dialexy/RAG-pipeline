"""
LLM generation layer. Takes retrieved chunks and a query, returns an answer.
"""

import ollama
from config import GenerationConfig


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    raise NotImplementedError


def generate(query: str, chunks: list[dict], cfg: GenerationConfig) -> str:
    """
    Call the local Ollama model with the retrieved context and return the answer string.
    Streams from ollama.chat() using cfg.model (default: qwen2.5:14b).
    """
    raise NotImplementedError
