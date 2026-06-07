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
from collections import defaultdict
from statistics import quantiles
from src.logger import get_logger
from typing import Any
from pathlib import Path
from textwrap import dedent
from src.vector_store import load_index, load_chunk_corpus
from src.retrieval import retrieve
from src.generation import generate, ABSTENTION_RESPONSE
from config import PipelineConfig, GenerationConfig, RAW_DIR

logger = get_logger(__name__)

QUESTION_TYPES = ["factual", "paraphrased", "inferential", "multi_hop"]

QUESTION_PROMPTS = {
    "factual": """Read the following text and generate one factual question
whose answer is clearly stated in the text.
Respond with only the question, nothing else.

Text: {window}""",
    "paraphrased": """Read the following text and generate one factual question
whose answer is clearly stated in the text. 
The question must not use the same wording as the answer in the text — rephrase it.
Respond with only the question, nothing else.

Text: {window}""",
    "inferential": """Read the following text and generate one question that requires
reasoning or inference to answer — the answer should not be a phrase directly lifted
from the text, but should require interpreting or combining what is stated.
Respond with only the question, nothing else.

Text: {window}""",
    "multi_hop": """Read the following text and generate one question whose answer
requires combining two separate facts stated in different parts of the text.
Respond with only the question, nothing else.

Text: {window}""",
}


def _ollama_unload(model: str) -> None:
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
        logger.info("Unloaded model %s from VRAM", model)
    except Exception as e:
        logger.warning("Failed to unload model %s: %s", model, e)


def _ollama_unload_all() -> None:
    """Evict every model Ollama currently has loaded in VRAM."""
    try:
        loaded = ollama.ps().models
        for m in loaded:
            if m.model:
                _ollama_unload(m.model)
        if loaded:
            time.sleep(5)
    except Exception as e:
        logger.warning("Could not query/unload Ollama models: %s", e)


def _ollama_warmup(model: str, max_attempts: int = 5) -> None:
    """Confirm the model runner is stable before starting the eval loop."""
    for attempt in range(max_attempts):
        try:
            ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}])
            logger.info("Warmup OK for %s", model)
            return
        except Exception as e:
            wait = 30 * (attempt + 1)
            logger.warning("Warmup failed for %s, retrying in %ds: %s", model, wait, e)
            time.sleep(wait)
    raise RuntimeError(f"Model {model} failed to warm up after {max_attempts} attempts")


def _ollama_chat_with_retry(
    model: str, messages: list[dict], max_retries: int = 10
) -> str:
    for attempt in range(max_retries):
        try:
            return (
                ollama.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": 0.0, "seed": 0},
                ).message.content
                or ""
            )
        except (
            ConnectionError,
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            ollama.ResponseError,
        ) as e:
            if attempt < max_retries - 1:
                # Give the runner longer to recover from a crash
                is_runner_crash = isinstance(
                    e, ollama.ResponseError
                ) and "unexpectedly stopped" in str(e)
                wait = 90 if is_runner_crash else min(5 * (2**attempt), 60)
                logger.warning(
                    "Ollama error, retrying in %ds (attempt %d/%d): %s",
                    wait,
                    attempt + 1,
                    max_retries,
                    e,
                )
                time.sleep(wait)
            else:
                raise RuntimeError("Ollama service unavailable after retries") from e
    return ""


def generate_qa_pairs(
    raw_dir: Path, n: int, cfg: GenerationConfig, seed: int = 42
) -> list[dict]:
    """Sample n docs from raw_dir, prompt the LLM for one factual question per doc,
    and return  (question, doc_id) pairs."""

    rng = random.Random(seed)
    text_files = list(raw_dir.rglob("*.txt"))
    rand_sample = rng.sample(text_files, min(n, len(text_files)))

    qa_pairs = []

    for i, file in enumerate(rand_sample):
        question_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]

        text = file.read_text(encoding="utf-8")
        doc_id = str(file.relative_to(raw_dir))

        max_start = max(0, len(text) - 2000)
        start = rng.randint(0, max_start)
        window = text[start : start + 2000]

        prompt = QUESTION_PROMPTS[question_type].format(window=window)

        question = _ollama_chat_with_retry(
            cfg.model, [{"role": "user", "content": prompt}]
        )

        qa_pairs.append(
            {
                "question": question,
                "relevant_doc_id": doc_id,
                "question_type": question_type,
            }
        )

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


def _decompose_claims(answer: str, cfg: GenerationConfig) -> list[str]:
    """split an answer into atomic claims, one per line"""
    prompt = dedent(f"""
    Split the following answer into individual atomic claims.
    Each claim should be a single self-contained factual assertion.
    Do not include source citations or hedging phrases like "according to".
    Respond with only the claims, one per line, nothing else.

    Answer: {answer}
    """)

    raw = _ollama_chat_with_retry(cfg.model, [{"role": "user", "content": prompt}])
    claims = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    return claims


def _verify_claim(claim: str, context: str, cfg: GenerationConfig) -> bool:
    """Return True if the claim is directly supported by the context."""
    prompt = dedent(f"""
    Context: {context}

    Claim: {claim}

    Is this claim directly supported by the context above?
    Respond with only "yes" or "no".
    """)

    response = _ollama_chat_with_retry(cfg.model, [{"role": "user", "content": prompt}])
    return "yes" in response.lower()


def faithfulness_score(
    answer: str, source_chunks: list[str], cfg: GenerationConfig
) -> float:
    """Decompose answer into atomic claims, verify each against context."""
    context = "\n".join(source_chunks)
    claims = _decompose_claims(answer, cfg)

    if not claims:
        logger.warning(
            "Claim decomposition returned no claims for answer: %r", answer[:100]
        )
        return 0.0

    verified = [_verify_claim(claim, context, cfg) for claim in claims]
    score = sum(verified) / len(verified)
    logger.debug("Faithfulness: %d/%d claims supported", sum(verified), len(verified))
    return score


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if len(latencies) < 2:
        v = latencies[0] if latencies else 0.0
        return {"p50": round(v, 3), "p95": round(v, 3)}
    q = quantiles(latencies, n=100)
    return {"p50": round(q[49], 3), "p95": round(q[94], 3)}


def run_evaluation(
    pipeline_cfg: PipelineConfig,
    qa_pairs: list[dict[str, Any]],
    judge_cfg: GenerationConfig | None = None,
) -> dict:
    """
    Run the full eval suite over a list of {"question": str, "relevant_doc_id": str}
    and return aggregated metrics.
    """

    _ollama_unload_all()
    _ollama_warmup(pipeline_cfg.generation.model)

    recall_scores = []
    mrr_scores = []
    hit3_scores = []
    faithfulness_scores = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    type_hit3: defaultdict[str, list] = defaultdict(list)
    type_recall: defaultdict[str, list] = defaultdict(list)

    abstention_count: int = 0

    effective_judge = judge_cfg if judge_cfg is not None else pipeline_cfg.generation

    collection = load_index(pipeline_cfg)
    corpus = load_chunk_corpus(collection)

    # Phase 1: retrieval + generation (14b model stays loaded)
    answers = []
    source_chunks_list = []
    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        relevant_doc_id = qa_pair["relevant_doc_id"]
        qtype = qa_pair.get("question_type", "factual")

        t0 = time.perf_counter()
        results, candidates = retrieve(
            question, collection, corpus, pipeline_cfg, return_candidates=True
        )
        retrieval_latencies.append(time.perf_counter() - t0)

        relevant_ids = {relevant_doc_id}
        candidate_ids = [c["metadata"]["parent_id"] for c in candidates]
        retrieved_ids = [r["metadata"]["parent_id"] for r in results]

        recall_score = recall_at_k(candidate_ids, relevant_ids, k=10)
        hit3_score = 1 if relevant_doc_id in retrieved_ids else 0

        recall_scores.append(recall_score)
        mrr_scores.append(mean_reciprocal_rank(candidate_ids, relevant_ids))
        hit3_scores.append(hit3_score)

        type_recall[qtype].append(recall_score)
        type_hit3[qtype].append(hit3_score)

        t0 = time.perf_counter()
        answers.append(generate(question, results, pipeline_cfg.generation))
        generation_latencies.append(time.perf_counter() - t0)
        source_chunks_list.append([r["text"] for r in results])

    # Evict generation model before loading the judge to avoid CUDA OOM
    _ollama_unload(pipeline_cfg.generation.model)

    # Calling latency helper after list are populated
    retrieval_stats = _latency_stats(retrieval_latencies)
    generation_stats = _latency_stats(generation_latencies)

    # Phase 2: faithfulness scoring (judge model loads once, 14b is killed)
    abstention_str = ABSTENTION_RESPONSE.lower()
    for answer, source_chunks in zip(answers, source_chunks_list):
        if abstention_str in answer.lower():
            abstention_count += 1
            continue

        score = faithfulness_score(answer, source_chunks, effective_judge)
        faithfulness_scores.append(score)

    mean_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else None
    )

    return {
        "recall@10_candidates": sum(recall_scores) / len(recall_scores),
        "mrr@10_candidates": sum(mrr_scores) / len(mrr_scores),
        "hit@3": sum(hit3_scores) / len(hit3_scores),
        "faithfulness": mean_faithfulness,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_count / len(qa_pairs),
        "n_evaluated": len(qa_pairs),
        "retrieval_p50_s": retrieval_stats["p50"],
        "retrieval_p95_s": retrieval_stats["p95"],
        "generation_p50_s": generation_stats["p50"],
        "generation_p95_s": generation_stats["p95"],
        "per_type": {
            qtype: {
                "n": len(type_hit3[qtype]),
                "hit@3": sum(type_hit3[qtype]) / len(type_hit3[qtype]),
                "recall@10": sum(type_recall[qtype]) / len(type_recall[qtype]),
            }
            for qtype in type_hit3
        },
    }


if __name__ == "__main__":
    import json
    import sys
    from datetime import datetime

    cfg = PipelineConfig()
    judge_cfg = GenerationConfig(model="qwen2.5:32b-instruct-q4_K_M")

    qa_set_path = Path(__file__).parent / "qa_set.json"

    if qa_set_path.exists() and "--regenerate" not in sys.argv:
        qa_pairs = json.loads(qa_set_path.read_text())
        logger.info("Loaded %d QA pairs from %s", len(qa_pairs), qa_set_path)
    else:
        qa_pairs = generate_qa_pairs(RAW_DIR, 100, judge_cfg, seed=42)
        qa_set_path.write_text(json.dumps(qa_pairs, indent=2))
        logger.info("Generated and saved %d QA pairs to %s", len(qa_pairs), qa_set_path)
    results = run_evaluation(cfg, qa_pairs, judge_cfg=judge_cfg)
    print(results)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S %d-%m-%y")
    results_path = results_dir / f"{timestamp}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {results_path}")
