"""
Evaluation: retrieval quality and end-to-end answer quality.

Metrics to implement:
  - Retrieval: Recall@k, MRR, NDCG
  - Answer: faithfulness (do answers stay grounded in retrieved chunks?),
            answer relevance, context precision/recall (RAGAs-style)
"""
from typing import Any


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    raise NotImplementedError


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    raise NotImplementedError


def faithfulness_score(answer: str, source_chunks: list[str]) -> float:
    """Check how many answer claims are grounded in the retrieved context."""
    raise NotImplementedError


def run_evaluation(pipeline_cfg, qa_pairs: list[dict[str, Any]]) -> dict:
    """
    Run the full eval suite over a list of {"question": str, "answer": str, "relevant_ids": list}
    and return aggregated metrics.
    """
    raise NotImplementedError
