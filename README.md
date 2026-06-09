# RAG Pipeline

A from-scratch Retrieval-Augmented Generation pipeline over a 10,000-article Wikipedia corpus. Built to study where naive RAG breaks and what targeted fixes look like: dense search alone undershoots on recall and conflates similarity with relevance. This pipeline adds semantic chunking, BM25 fusion, query expansion, cross-encoder reranking, and reranker-score abstention to address each failure mode, with an evaluation harness designed so the benchmark numbers are actually trustworthy.

## Architecture

Two separate pipelines: one that runs once to build the index, and one that runs per query.

```
INDEX TIME
──────────────────────────────────────────────────────
Wikipedia corpus (10,000 articles)
        │
        ▼
  Ingestion              fetch_corpus() → data/raw/*.txt
        │
        ▼
  Chunking               semantic split (std_multiplier=1.5)
        │                token-aware, hard-split fallback at chunk_size_tokens=512
        │                → ~342k chunks across the corpus
        ▼
  Embedding              BAAI/bge-large-en-v1.5 → 1024-dim vectors
        │
        ▼
  Vector Store           ChromaDB (persistent, cosine distance, batched upserts)
                         index config stored as collection metadata


QUERY TIME
──────────────────────────────────────────────────────
User query
        │
        ▼
  Query Expansion        LLM (qwen2.5:14b) generates 2 paraphrase variants
        │                → [original, variant_1, variant_2]
        ▼
  Batched Embedding      all variants embedded in one bge-large call
        │                (query instruction prefix applied)
        │
        ├──────────────────────┐
        ▼                      ▼
  Dense Search           BM25 Search
  (per variant,          (per variant,
   ChromaDB ANN)          chunk corpus)
        │                      │
        └──────────┬───────────┘
                   ▼
            RRF Fusion             rank-based merge across all variant lists
                   │
                   ▼
          Cross-Encoder Rerank    BAAI/bge-reranker-large, top_n=3
                   │               (neighbour-stitched: [prev, current, next])
                   ▼
          Reranker Score Gate     top score < threshold → abstain
                   │
                   ▼
            Generation            Ollama (qwen2.5:14b, local)
                   │
                   ▼
             Answer + Sources
```

## Tech Stack

| Component | Library / Model |
|---|---|
| Embeddings | `sentence-transformers`, `BAAI/bge-large-en-v1.5` (1024-dim, asymmetric) |
| Vector store | `chromadb` (persistent, cosine) |
| Sparse retrieval | `rank-bm25` (chunk-level corpus) |
| Query expansion | Ollama, `qwen2.5:14b` |
| Reranker | `BAAI/bge-reranker-large` (cross-encoder, GPU with CPU fallback) |
| Generation | Ollama, `qwen2.5:14b` (local) |
| Eval judge | Ollama, `qwen2.5:32b-instruct-q4_K_M` (judge only, never generation) |
| Corpus | Hugging Face `datasets`, Wikipedia |
| API | `fastapi`, `uvicorn` |

## Benchmark Results

All numbers below are on a fixed gold QA set of `n=100` questions, evaluated with `qwen2.5:32b-instruct-q4_K_M` as judge and `qwen2.5:14b` for generation. Metrics:

- **Recall@10**: true recall measured on the pre-rerank candidate pool (`recall@10_candidates`).
- **MRR**: mean reciprocal rank over the same candidate pool.
- **Hit@3**: relevant chunk present in the top 3 after reranking.
- **Faithfulness**: fraction of generated answers grounded in retrieved context, with abstentions excluded from the denominator and reported separately.

### Main benchmark

| Run | Recall@10 | MRR | Hit@3 | Faithfulness | Notes |
|---|---|---|---|---|---|
| True baseline (all critical fixes: token chunking, cosine, normalized, fixed QA set) | 0.92 | 0.793 | 0.89 | 0.846 | First fully comparable run |
| + bge-large embedding upgrade | 0.94 | 0.869 | 0.92 | 0.895 | |
| Semantic threshold 1.5 selected (342k chunks) | 0.93 | 0.851 | 0.96 | 0.899 | Best Hit@3 across the sweep |
| **Threshold confirmed @ 1.5** | **0.940** | **0.844** | **0.960** | **0.922** | Last comparable clean GPU run |
| **Harder QA set (mixed question types)** | **0.940** | **0.862** | **0.980** | **0.963** | Superseded by final row |
| **Final (clean index, claim decomposition faithfulness)** | **0.960** | **0.784** | **0.990** | **0.932** | **Definitive result — see below** |

The **final row** is the definitive result. It uses the same mixed question set (factual, paraphrased, inferential, multi-hop) with claim decomposition faithfulness and a clean index rebuilt after fixing a `chunk_recursive` duplication bug that had inflated the index by ~107k duplicate chunks (342,188 → 234,586). It was CPU-reranked due to VRAM pressure; quality metrics are valid, latency is not representative. Failure breakdown: 0 retrieval failures, 19 generation failures — all faithfulness failures are generation-side.

The **threshold-confirmed @ 1.5** run is the last fully comparable clean **GPU** run, and is the reference point for any latency comparison on the main pipeline.

#### Per-type breakdown (harder QA set)

| Type | Hit@3 | Recall@10 |
|---|---|---|
| factual | 0.96 | 0.92 |
| paraphrased | 1.00 | 1.00 |
| inferential | 0.96 | 0.92 |
| multi_hop | 1.00 | 0.92 |

Paraphrased and multi-hop sit at ceiling, most likely because query expansion closes the lexical gap that those question types otherwise expose.

> **Excluded runs.** Several earlier runs predate the methodology fixes (`n=20` samples, no persisted gold QA set, and `Recall@10` mislabelled, when it was actually Hit@3). These are **pre-methodology-fix** and are not comparable to the table above, so they are excluded rather than shown with misleading labels.

### Ablation

Component ablation over the fixed QA set, toggling `use_hybrid`, `use_reranker`, and `use_query_expansion`. Reported in its own table because the configs are not iterative improvements; they isolate each component's contribution. GPU-reranked, so retrieval latencies here are representative.

| Config | Recall@10 | MRR | Hit@3 | Faithfulness | Abstention | Retrieval p50 |
|---|---|---|---|---|---|---|
| baseline (dense only) | 0.85 | 0.755 | 0.80 | 0.954 | 0.21 | 0.011s |
| + hybrid (BM25) | 0.93 | 0.847 | 0.90 | 0.970 | 0.11 | 0.891s |
| + reranker | 0.85 | 0.755 | 0.83 | 0.972 | 0.21 | 0.225s |
| + expansion | 0.86 | 0.744 | 0.79 | 0.952 | 0.22 | 3.942s |
| **full** | **0.93** | **0.853** | **0.96** | **0.975** | **0.13** | **7.751s** |

Reading the ablation:

- **Hybrid is the single biggest contributor**: +8pp Recall@10 and +10pp Hit@3 over dense-only. BM25 recovers the exact-match queries dense search misses.
- **Query expansion is net-negative in isolation** (it widens the candidate pool with noise when nothing else filters it), but on top of hybrid + reranker it adds +6pp Hit@3; the reranker absorbs the extra candidates and keeps the good ones.
- **The expansion LLM call dominates retrieval latency** (3.94s p50 on its own), which is the cost of the full config's 7.75s p50.

### Unanswerable / distractor evaluation

A separate `n=100` set of out-of-corpus questions, generated by streaming Wikipedia past the indexed 10,000 articles. This measures whether the pipeline correctly abstains instead of fabricating answers for questions the corpus cannot support.

| Metric | Value |
|---|---|
| Abstention accuracy | 0.96 |
| False answer rate | 0.04 |

The dominant failure mode is **semantic false positives**: population/statistics queries retrieve plausible same-domain chunks from the *wrong* article (e.g. answering a population question with a different town's figure), which then passes the gate and gets answered. This run was CPU-reranked; a GPU run may differ marginally.

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally. The 32B judge (~20GB) and the reranker (~1GB) coexist with careful phase separation during eval. On a 12 GiB card, set `OLLAMA_GPU_OVERHEAD=3221225472` in the Ollama service environment to reserve 3 GiB of headroom for the reranker and embedding model.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pull the models:

```bash
ollama pull qwen2.5:14b                       # generation + query expansion
ollama pull qwen2.5:32b-instruct-q4_K_M       # eval judge only
```

## Usage

**Index**: run once (or whenever the corpus changes):

```bash
python -m main index
```

This fetches the Wikipedia corpus, chunks, embeds, and persists the ChromaDB index to `.chroma/`. The corpus hash is combined with a config fingerprint, so changing the chunking or embedding config triggers a rebuild automatically. Pass `--force` to reindex unconditionally.

**Query:**

```bash
python -m main query "What is the speed of light?"
python -m main query "Who invented the printing press?" --model qwen2.5:14b
```

The generated answer is printed to stdout. Sources with reranker scores are available via the Python API:

```python
from config import PipelineConfig
from src.pipeline import build_pipeline, query_pipeline

cfg = PipelineConfig()

build_pipeline(cfg)            # index time: run once
result = query_pipeline("What is the speed of light?", cfg)

print(result["answer"])
for src in result["sources"]:
    print(src["metadata"]["fileName"], src.get("reranked_score"))
```

**Serve**: start the FastAPI app (POST `/query`, GET `/health`):

```bash
python -m main serve --host 0.0.0.0 --port 8000
```

## Evaluation

Three independent eval entry points, each writing timestamped JSON to `eval/results/`:

```bash
python -m eval.evaluate        # core retrieval + generation quality
python -m eval.ablation        # component ablation
python -m eval.unanswerable    # out-of-corpus abstention
```

- **`eval.evaluate`**: runs the gold QA set (`eval/qa_set.json`) through the full pipeline and reports Recall@10 / MRR on pre-rerank candidates, Hit@3, faithfulness, abstention rate, per-type breakdown, and p50/p95 retrieval and generation latency. Two-phase: all generation (14B) first, then all judging (32B), with a model unload between phases to avoid CUDA OOM. Pass `--regenerate` to rebuild the gold QA set instead of loading the persisted one.
- **`eval.ablation`**: sweeps `use_hybrid`, `use_reranker`, and `use_query_expansion` over the same fixed QA set and emits a `{config → recall, MRR, Hit@3, faithfulness, latency}` table.
- **`eval.unanswerable`**: streams Wikipedia past index 10,000, generates 100 out-of-corpus questions, and measures abstention accuracy and false-answer rate against questions the corpus cannot answer.

## Tests

```bash
pytest
```

26 tests across 4 files, all passing. They cover pure functions only (no Ollama, ChromaDB, or HuggingFace dependencies), so they run fast and offline: `recall_at_k`, `mean_reciprocal_rank`, `_latency_stats` (evaluate), `reciprocal_rank_fusion`, `bm25_search` (retrieval), `file_hash`, `corpus_hash`, `config_fingerprint` (pipeline), the fixed/recursive chunking strategies, and `build_config` mutation safety (ablation). Semantic chunking is excluded because it requires loading the embedding model.

## Project Structure

```
main.py                CLI entry point (index / query / serve subcommands)
api.py                 FastAPI app (POST /query, GET /health)
config.py              Dataclass-based config for all pipeline stages
requirements.txt       Dependencies
src/
  models.py            Document dataclass
  ingestion.py         Corpus download and document iteration (Hugging Face datasets)
  chunking.py          Semantic, recursive, and fixed-size chunking strategies
  embedding.py         bge-large batched encoding (query-instruction aware)
  vector_store.py      ChromaDB build/search, chunk corpus + neighbour fetch
  retrieval.py         query expansion, BM25, RRF fusion, cross-encoder reranking
  generation.py        Ollama prompt construction, inference, abstention handling
  pipeline.py          build_pipeline() and query_pipeline() orchestration
  logger.py            Logging setup
eval/
  evaluate.py              core eval loop (Recall@10, MRR, Hit@3, faithfulness, latency)
  ablation.py              component ablation harness
  unanswerable.py          out-of-corpus abstention eval
  qa_set.json              persisted gold QA set (mixed question types)
  qa_set_factual_only.json archived factual-only gold set
  unanswerable_set.json    persisted out-of-corpus question set
  results/                 timestamped run outputs (JSON)
tests/
  test_evaluate.py     recall_at_k, mean_reciprocal_rank, _latency_stats
  test_retrieval.py    reciprocal_rank_fusion, bm25_search
  test_pipeline.py     file_hash, corpus_hash, config_fingerprint
  test_chunking.py     fixed / recursive chunking strategies
  test_ablation.py     build_config mutation safety
data/
  raw/                 downloaded .txt documents
.chroma/               persistent ChromaDB vector index
```

## Configuration

All knobs live in `config.py` as dataclasses:

| Config | Key fields | Defaults |
|---|---|---|
| `ChunkConfig` | `strategy`, `chunk_size`, `chunk_size_tokens`, `chunk_overlap`, `semantic_split_std_multiplier` | `semantic`, 1500, 512, 64, 1.5 |
| `EmbeddingConfig` | `model_name`, `batch_size`, `query_instruction` | `BAAI/bge-large-en-v1.5`, 64, `"Represent this sentence for searching relevant passages: "` |
| `RetrievalConfig` | `top_k`, `rerank_top_n`, `use_hybrid`, `use_reranker`, `use_query_expansion`, `reranker_score_threshold` | 10, 3, True, True, True, 0.0 |
| `GenerationConfig` | `model`, `max_tokens`, `temperature` | `qwen2.5:14b`, 1024, 0.0 |

### Chunking

The semantic chunker splits documents at sentence-embedding distance peaks. `semantic_split_std_multiplier=1.5` splits at the top tail of sentence-boundary distances per document, producing ~342k chunks across the 10k corpus, the best Hit@3 of the swept values (1.5, 2.5, 3.0, 3.5; everything above 2.5 plateaued at ~423–432k chunks with diminishing returns). `chunk_size_tokens=512` is the bge-large token limit and bounds chunk length; `chunk_size=1500` (chars) is retained only as a hard-split fallback. Because bge-large is asymmetric, the query-instruction prefix is applied to queries but not passages.

### Reranker score gating

If the top reranker score falls below `reranker_score_threshold`, the pipeline short-circuits to an abstention response before generation, rather than answering from weak context. The default threshold is `0.0` (effectively off until tuned from the observed score distribution), but the gate is what drives the unanswerable-set abstention behaviour once raised.

## Retrieval Design

Naive dense-only RAG fails in predictable ways, and each pipeline stage targets one:

**1. Vocabulary mismatch.** An exact-keyword query ("Battle of Hastings 1066") can score poorly against dense neighbours if the corpus phrases it differently. BM25 handles exact-term overlap natively. Dense and BM25 lists (one pair per query variant) are merged with **Reciprocal Rank Fusion**, which is rank-based, so it sidesteps the score-scale incompatibility between cosine similarity and BM25 scores. The ablation confirms this is the single largest quality lever.

**2. Phrasing brittleness.** A single phrasing of a question only probes one region of embedding space. **Query expansion** asks the LLM for two paraphrases; all three variants are embedded in one batched call and retrieved independently before fusion. In isolation this adds noise, but combined with the reranker it lifts Hit@3, most visibly on paraphrased and multi-hop questions.

**3. Similarity ≠ relevance.** Top-*k* by cosine distance is not top-*k* by answer quality. A **cross-encoder reranker** scores each query–chunk pair jointly, attending to fine-grained interactions. It's expensive, so it runs only over the RRF-fused candidates and selects `rerank_top_n=3`. Selected chunks are neighbour-stitched into strict `[prev, current, next]` order (deduplicated) to restore local context for generation.

**4. Unsupported questions.** When the best candidate is still weak, answering is the wrong move. The **reranker score gate** abstains before generation. On the out-of-corpus set this yields 96% abstention accuracy at a 4% false-answer rate.

## Known Limitations

- **BM25 index is not persisted**: it is rebuilt in-memory from the chunk corpus on load. Cached per-process via `lru_cache`, but cold on every restart.
- **Chunk deduplication is disabled at scale**: the O(n²) exact-dedup approach is unusable above ~100k chunks. A FAISS approximate-nearest-neighbour rewrite is tracked separately.
- **Semantic false positives on statistics queries**: the dominant unanswerable-set failure is that population/figure questions retrieve a plausible same-domain chunk from the wrong article and answer with the wrong number. Tighter gating or entity-aware filtering is the likely fix.
- **Shared config mutation in the API**: `/query` mutates the process-wide `cfg` when an override model is passed, which is not thread-safe. A per-request copy is needed.
- **Faithfulness is answer-level, not claim-level**: it judges the whole answer rather than decomposing it into atomic claims, so a partially-grounded answer is scored as a single pass/fail.
