import json
import sys
import random
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from config import PipelineConfig, GenerationConfig
from src.pipeline import query_pipeline
from src.generation import ABSTENTION_RESPONSE
from src.vector_store import load_index, load_chunk_corpus
from src.models import Document
from src.ingestion import CORPUS_SIZE
from eval.evaluate import _ollama_chat_with_retry
from src.logger import get_logger

logger = get_logger(__name__)

UNANSWERABLE_SET_PATH = Path(__file__).parent / "unanswerable_set.json"


def generate_unanswerable_qa_pairs(
    n: int,
    cfg: GenerationConfig,
    seed: int = 42,
) -> list[dict]:
    """
    Stream wikimedia/wikipedia 20231101.en, skip first CORPUS_SIZE articles,
    sample n articles from the remainder, generate one question per article.
    Returns list of {"question": str, "source_title": str}.
    No relevant_doc_id — these are unanswerable by design.
    """
    rng = random.Random(seed)

    logger.info("Streaming Wikipedia dataset, skipping first %d articles", CORPUS_SIZE)
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )

    pool: list[dict] = []
    for i, item in enumerate(ds):
        if i < CORPUS_SIZE:
            continue
        pool.append({"title": item["title"], "text": item["text"]})
        if len(pool) >= n * 10:
            break

    sampled = rng.sample(pool, min(n, len(pool)))
    logger.info("Sampled %d articles from post-corpus Wikipedia", len(sampled))

    qa_pairs = []
    for item in sampled:
        text = item["text"]
        max_start = max(0, len(text) - 2000)
        start = rng.randint(0, max_start)
        window = text[start : start + 2000]

        prompt = f"""Read the following text and generate one factual question
whose answer is clearly stated in the text.
Respond with only the question, nothing else.

Text: {window}"""

        question = _ollama_chat_with_retry(
            cfg.model, [{"role": "user", "content": prompt}]
        )
        qa_pairs.append({"question": question, "source_title": item["title"]})

    return qa_pairs


def evaluate_unanswerable(
    pipeline_cfg: PipelineConfig,
    unanswerable_pairs: list[dict],
    collection=None,
    corpus: list[Document] | None = None,
) -> dict:
    """
    Run each unanswerable question through query_pipeline.
    Measure correct abstentions vs false answers.
    """
    correct_abstentions = 0
    false_answers = []
    abstention_str = ABSTENTION_RESPONSE.lower()

    for pair in unanswerable_pairs:
        result = query_pipeline(
            pair["question"], pipeline_cfg, collection=collection, corpus=corpus
        )
        answer = result["answer"]

        if abstention_str in answer.lower():
            correct_abstentions += 1
        else:
            false_answers.append(
                {
                    "question": pair["question"],
                    "source_title": pair["source_title"],
                    "answer": answer,
                }
            )

    n = len(unanswerable_pairs)
    return {
        "n_unanswerable": n,
        "correct_abstentions": correct_abstentions,
        "abstention_accuracy": correct_abstentions / n,
        "false_answer_rate": len(false_answers) / n,
        "false_answers": false_answers,
    }


if __name__ == "__main__":
    cfg = PipelineConfig()
    judge_cfg = GenerationConfig(model="qwen2.5:32b-instruct-q4_K_M")

    if UNANSWERABLE_SET_PATH.exists() and "--regenerate" not in sys.argv:
        unanswerable_pairs = json.loads(UNANSWERABLE_SET_PATH.read_text())
        logger.info(
            "Loaded %d unanswerable pairs from %s",
            len(unanswerable_pairs),
            UNANSWERABLE_SET_PATH,
        )
    else:
        unanswerable_pairs = generate_unanswerable_qa_pairs(100, judge_cfg, seed=42)
        UNANSWERABLE_SET_PATH.write_text(json.dumps(unanswerable_pairs, indent=2))
        logger.info(
            "Generated and saved %d unanswerable pairs to %s",
            len(unanswerable_pairs),
            UNANSWERABLE_SET_PATH,
        )

    collection = load_index(cfg)
    corpus = load_chunk_corpus(collection)

    results = evaluate_unanswerable(
        cfg, unanswerable_pairs, collection=collection, corpus=corpus
    )

    print(f"\nUnanswerable eval over {results['n_unanswerable']} questions")
    print(f"  Abstention accuracy : {results['abstention_accuracy']:.3f}")
    print(f"  False answer rate   : {results['false_answer_rate']:.3f}")
    print(f"  False answers       : {len(results['false_answers'])}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S %d-%m-%y")
    results_path = results_dir / f"unanswerable_{timestamp}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {results_path}")
