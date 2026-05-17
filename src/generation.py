"""
LLM generation layer. Takes retrieved chunks and a query, returns an answer.
"""

import ollama
from textwrap import dedent
from config import GenerationConfig


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

    Answer based only on the context above.
    If the answer is not in the context,
    say "I don't have enough information to answer that."
    and explain what was given.
    """)

    return final_prompt


def generate(query: str, chunks: list[dict], cfg: GenerationConfig) -> str:
    """
    Call the local Ollama model with the retrieved context and return the answer string.
    Streams from ollama.chat() using cfg.model (default: qwen2.5:14b).
    """

    prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
    )

    return response.message.content or ""
