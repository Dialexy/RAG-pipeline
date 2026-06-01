import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from config import GenerationConfig, PipelineConfig, RAW_DIR
from eval.evaluate import generate_qa_pairs, run_evaluation
from src.logger import get_logger

logger = get_logger(__name__)

ABLATION_CONFIGS: list[dict] = [
    {
        "name": "baseline",
        "use_hybrid": False,
        "use_reranker": False,
        "use_query_expansion": False,
    },
    {
        "name": "+hybrid",
        "use_hybrid": True,
        "use_reranker": False,
        "use_query_expansion": False,
    },
    {
        "name": "+reranker",
        "use_hybrid": False,
        "use_reranker": True,
        "use_query_expansion": False,
    },
    {
        "name": "+expansion",
        "use_hybrid": False,
        "use_reranker": False,
        "use_query_expansion": True,
    },
    {
        "name": "full",
        "use_hybrid": True,
        "use_reranker": True,
        "use_query_expansion": True,
    },
]


def build_config(base: PipelineConfig, variant: dict) -> PipelineConfig:
    new_retrieval = replace(
        base.retrieval,
        use_hybrid=variant["use_hybrid"],
        use_reranker=variant["use_reranker"],
        use_query_expansion=variant["use_query_expansion"],
    )
    return replace(base, retrieval=new_retrieval)


def print_quality_table(rows: list[dict]) -> None:
    header = (
        f"{'Config':<14} {'Recall@10':>10} {'MRR':>8} {'Hit@3':>8} {'Faithful':>10}"
    )
    divider = "-" * len(header)
    print(header)
    print(divider)
    for r in rows:
        faithful = (
            f"{r['faithfulness']:.3f}" if r["faithfulness"] is not None else "   N/A"
        )
        print(
            f"{r['name']:<14} {r['recall@10_candidates']:>10.3f}"
            f" {r['mrr@10_candidates']:>8.3f}"
            f" {r['hit@3']:>8.3f}"
            f" {faithful:>10}"
        )


def print_latency_table(rows: list[dict]) -> None:
    header = (
        f"{'Config':<14} {'Ret p50':>9} {'Ret p95':>9} {'Gen p50':>9} {'Gen p95':>9}"
    )
    divider = "-" * len(header)
    print(header)
    print(divider)
    for r in rows:
        print(
            f"{r['name']:<14}"
            f" {r['retrieval_p50_s']:>9.3f}"
            f" {r['retrieval_p95_s']:>9.3f}"
            f" {r['generation_p50_s']:>9.3f}"
            f" {r['generation_p95_s']:>9.3f}"
        )


def run_ablation(
    base_cfg: PipelineConfig,
    qa_pairs: list[dict],
    judge_cfg: GenerationConfig,
) -> list[dict]:
    rows = []
    for variant in ABLATION_CONFIGS:
        cfg = build_config(base_cfg, variant)
        logger.info("Running ablation variant: %s", variant["name"])
        results = run_evaluation(cfg, qa_pairs, judge_cfg=judge_cfg)
        rows.append({"name": variant["name"], **results})
    return rows


if __name__ == "__main__":
    base_cfg = PipelineConfig()
    judge_cfg = GenerationConfig(model="qwen2.5:32b-instruct-q4_K_M")

    qa_set_path = Path(__file__).parent / "qa_set.json"
    if qa_set_path.exists() and "--regenerate" not in sys.argv:
        qa_pairs = json.loads(qa_set_path.read_text())
        logger.info("Loaded %d QA pairs from %s", len(qa_pairs), qa_set_path)
    else:
        qa_pairs = generate_qa_pairs(RAW_DIR, 100, judge_cfg, seed=42)
        qa_set_path.write_text(json.dumps(qa_pairs, indent=2))
        logger.info("Generated and saved %d QA pairs to %s", len(qa_pairs), qa_set_path)

    rows = run_ablation(base_cfg, qa_pairs, judge_cfg)

    print("\nQuality")
    print_quality_table(rows)
    print("\nLatency (seconds)")
    print_latency_table(rows)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S %d-%m-%y")
    results_path = results_dir / f"ablation_{timestamp}.json"
    results_path.write_text(json.dumps(rows, indent=2))
    print(f"\nResults written to {results_path}")
