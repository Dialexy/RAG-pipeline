"""
Evaluation: retrieval quality and end-to-end answer quality.

Metrics to implement:
  - Retrieval: Recall@k, MRR, NDCG
  - Answer: faithfulness (do answers stay grounded in retrieved chunks?),
            answer relevance, context precision/recall (RAGAs-style)
"""

import ollama
import random
import re
from typing import Any
from pathlib import Path
from textwrap import dedent
from src.vector_store import load_index
from src.retrieval import retrieve
from src.generation import generate
from src.ingestion import iter_documents
from src.models import Document
from config import PipelineConfig, GenerationConfig, RAW_DIR


def generate_qa_pairs(raw_dir: Path, n: int, cfg: GenerationConfig) -> list[dict]:
    """Sample n docs from raw_dir, prompt the LLM for one factual question per doc,
    and return  (question, doc_id) pairs."""

    text_files = list(raw_dir.rglob("*.txt"))
    rand_sample = random.sample(text_files, min(n, len(text_files)))

    qa_pairs = []

    for file in rand_sample:
        text = file.read_text(encoding="utf-8")
        doc_id = str(file.relative_to(raw_dir))

        max_start = max(0, len(text) - 2000)
        start = random.randint(0, max_start)
        window = text[start:start + 2000]

        prompt = f"""
                Read the following text and generate one factual question
                whose answer is clearly stated in the text.
                Respond with only the question, nothing else.

                Text: {window}
                """

        question = (
            ollama.chat(
                model=cfg.model, messages=[{"role": "user", "content": prompt}]
            ).message.content
            or ""
        )

        qa_pairs.append({"question": question, "relevant_doc_id": doc_id})

    return qa_pairs


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k_ids = retrieved_ids[:k]
    hits = len(set(top_k_ids) & relevant_ids)

    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, id in enumerate(retrieved_ids, start=1):
        if id in relevant_ids:
            return 1.0 / rank
    return 0.0


def faithfulness_score(
    answer: str, source_chunks: list[str], cfg: GenerationConfig
) -> float:
    """Check how many answer claims are grounded in the retrieved context."""
    context = "\n".join(source_chunks)

    prompt = dedent(f"""
    You are evaluating if an answer is grounded in a context.

    Context: {context}

    Answer: {answer}

    For each claim in the answer, check if it is directly supported by the context.
    Respond with only a decimal between 0.0 and 1.0 representing the fraction of claims supported.
    0.0 means no claims are supported. 1.0 means all claims are supported.
    Only output the number, nothing else,
    """)

    ollama_output = ollama.chat(
        model=cfg.model, messages=[{"role": "user", "content": prompt}]
    )
    content_unstripped = ollama_output.message.content or ""
    content_stripped = content_unstripped.strip()

    match = re.search(r"\b\d*\.?\d+\b", content_stripped)
    if match:
        score = float(match.group())
        return max(0.0, min(1.0, score))
    return 0.0


def run_evaluation(
    pipeline_cfg: PipelineConfig, qa_pairs: list[dict[str, Any]]
) -> dict:
    """
    Run the full eval suite over a list of {"question": str, "answer": str, "relevant_ids": list}
    and return aggregated metrics.
    """
    recall_scores = []
    mrr_scores = []
    faithfulness_scores = []

    collection = load_index(pipeline_cfg)
    corpus = []
    batch_size = 5000
    offset = 0
    while True:
        batch = collection.get(include=["documents", "metadatas"], limit=batch_size, offset=offset)
        if not batch["ids"]:
            break
        for id_, text, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            corpus.append(Document(id=id_, text=text, metadata=meta))
        offset += batch_size

    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        relevant_doc_id = qa_pair["relevant_doc_id"]

        results = retrieve(question, collection, corpus, pipeline_cfg)

        retrieved_ids = [result["metadata"]["parent_id"] for result in results]
        relevant_ids = {relevant_doc_id}
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=10))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

        answer = generate(question, results, pipeline_cfg.generation)
        source_chunks = [r["text"] for r in results]
        faithfulness_scores.append(
            faithfulness_score(answer, source_chunks, pipeline_cfg.generation)
        )

    return {
        "recall@10": sum(recall_scores) / len(recall_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "n_evaluated": len(qa_pairs),
    }


if __name__ == "__main__":
    cfg = PipelineConfig()
    qa_pairs = generate_qa_pairs(RAW_DIR, 20, cfg.generation)
    results = run_evaluation(cfg, qa_pairs)
    print(results)
