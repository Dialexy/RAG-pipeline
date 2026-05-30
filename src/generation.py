"""
LLM generation layer. Takes retrieved chunks and a query, returns an answer.
"""

import time
import ollama
import httpx
from textwrap import dedent
from config import GenerationConfig
from .logger import get_logger

logger = get_logger(__name__)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""

    formatted_chunks = []

    for i, chunk in enumerate(chunks, start=1):
        formatted_string = f"source{i}:\n{chunk['text']}\n"
        formatted_chunks.append(formatted_string)

    all_formatted_chunks = "\n".join(formatted_chunks)

    final_prompt = dedent(f"""
    Context:
    {all_formatted_chunks}

    Question:
    {query}

    Answer the question using ONLY the sources provided above.
    Do not use any knowledge outside of the provided sources.
    If the answer cannot be found in the sources, respond with exactly:
    "I don't have enough information to answer that."
    Cite which source number supports your answer.
    """)

    return final_prompt


def generate(query: str, chunks: list[dict], cfg: GenerationConfig) -> str:
    """
    Call the local Ollama model with the retrieved context and return the answer string.
    Streams from ollama.chat() using cfg.model (default: qwen2.5:14b).
    """

    logger.info(
        "Generating answer for query: %r (model=%s, chunks=%d)",
        query,
        cfg.model,
        len(chunks),
    )
    prompt = build_prompt(query, chunks)

    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant that answers questions strictly based on provided context. Never use outside knowledge. If the answer isn't in the context, say so.",
        },
        {"role": "user", "content": prompt},
    ]
    options = {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=cfg.model, messages=messages, options=options)
            answer = response.message.content or ""
            logger.info("Generation complete: %d chars", len(answer))
            return answer

        except (ConnectionError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < max_retries - 1:
                wait = min(5 * (2**attempt), 60)
                logger.warning(
                    "Ollama connection dropped, retrying in %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Cannot connect to Ollama after %d attempts. Start it with 'ollama serve'",
                    max_retries,
                )
                raise RuntimeError("Ollama service unavailable") from e

        except ollama.ResponseError as e:
            logger.error("Ollama returned an error: %s", e)
            raise RuntimeError("Ollama service unavailable") from e

        except Exception as e:
            if "model" in str(e).lower() and cfg.model in str(e):
                logger.error(
                    "Model %r not found. Pull with 'ollama pull %s'",
                    cfg.model,
                    cfg.model,
                )
                raise RuntimeError(f"Model {cfg.model!r} not found") from e
            raise

    raise RuntimeError("Ollama service unavailable")
