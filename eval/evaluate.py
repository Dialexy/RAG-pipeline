"""
Evaluation: retrieval quality and end-to-end answer quality.

Metrics to implement:
  - Retrieval: Recall@k, MRR, NDCG
  - Answer: faithfulness (do answers stay grounded in retrieved chunks?),
            answer relevance, context precision/recall (RAGAs-style)
"""

import time
import ollama
import httpx
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


def _ollama_chat_with_retry(model: str, messages: list[dict], max_retries: int = 10) -> str:
    for attempt in range(max_retries):
        try:
            return ollama.chat(model=model, messages=messages).message.content or ""
        except (ConnectionError, httpx.RemoteProtocolError, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                wait = min(5 * (2 ** attempt), 60)
                print(f"Ollama disconnected, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise RuntimeError("Ollama service unavailable after retries") from e
    return ""


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
        window = text[start : start + 2000]

        prompt = f"""
                Read the following text and generate one factual question
                whose answer is clearly stated in the text.
                Respond with only the question, nothing else.

                Text: {window}
                """

        question = _ollama_chat_with_retry(cfg.model, [{"role": "user", "content": prompt}])

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

    content_stripped = _ollama_chat_with_retry(cfg.model, [{"role": "user", "content": prompt}]).strip()

    match = re.search(r"\b\d*\.?\d+\b", content_stripped)
    if match:
        score = float(match.group())
        return max(0.0, min(1.0, score))
    return 0.0


def run_evaluation(
    pipeline_cfg: PipelineConfig,
    qa_pairs: list[dict[str, Any]],
    judge_cfg: GenerationConfig | None = None,
) -> dict:
    """
    Run the full eval suite over a list of {"question": str, "answer": str, "relevant_ids": list}
    and return aggregated metrics.
    """
    recall_scores = []
    mrr_scores = []
    faithfulness_scores = []

    effective_judge = judge_cfg if judge_cfg is not None else pipeline_cfg.generation

    collection = load_index(pipeline_cfg)
    corpus = []
    batch_size = 5000
    offset = 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"], limit=batch_size, offset=offset
        )
        if not batch["ids"]:
            break
        for id_, text, meta in zip(
            batch["ids"], batch["documents"] or [], batch["metadatas"] or []
        ):
            corpus.append(Document(id=id_, text=text, metadata=dict(meta)))
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
            faithfulness_score(answer, source_chunks, effective_judge)
        )

    return {
        "recall@10": sum(recall_scores) / len(recall_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "n_evaluated": len(qa_pairs),
    }


if __name__ == "__main__":
    import json
    from datetime import datetime

    cfg = PipelineConfig()
    judge_cfg = GenerationConfig(model="qwen2.5:32b-instruct-q4_K_M")
    qa_pairs = generate_qa_pairs(RAW_DIR, 20, judge_cfg)
    results = run_evaluation(cfg, qa_pairs, judge_cfg=judge_cfg)
    print(results)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S %d-%m-%y")
    results_path = results_dir / f"{timestamp}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {results_path}")
