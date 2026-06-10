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
        │                → 234,586 chunks across the corpus
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
| Claim-decomposition faithfulness | 0.960 | 0.784 | 0.990 | 0.932 | Superseded — index held ~107k stale chunks (see below) |
| **Final (verified-clean index, 234,586 chunks)** | **0.970** | **0.909** | **1.000** | **0.938** | **Definitive result — see below** |

The **final row** is the definitive result, on the mixed question set (factual, paraphrased, inferential, multi-hop) with claim decomposition faithfulness. The previous "clean index" run turned out not to be clean: `build_index` upserted chunks without ever deleting, so the rebuild after the `chunk_recursive` duplication fix overwrote only the first 234,586 chunk ids and silently left 107,602 stale duplicates (31% of the collection) from the buggy generation in the retrieval pool. After fixing `build_index` to recreate the collection from scratch, the index was rebuilt to a verified 234,586 chunks and the eval re-run — every metric improved, as the stale duplicates had been competing with better chunks in RRF fusion and reranking (most visibly MRR, 0.784 → 0.909). This run was CPU-reranked due to VRAM pressure; quality metrics are valid, latency is not representative. Failure breakdown: 0 retrieval failures, 19 generation failures — all faithfulness failures are generation-side.

The **threshold-confirmed @ 1.5** run is the last fully comparable clean **GPU** run, and is the reference point for any latency comparison on the main pipeline.

#### Per-type breakdown (final run)

| Type | Hit@3 | Recall@10 |
|---|---|---|
| factual | 1.00 | 1.00 |
| paraphrased | 1.00 | 1.00 |
| inferential | 1.00 | 0.92 |
| multi_hop | 1.00 | 0.96 |

Hit@3 sits at ceiling for every question type on the clean index; the only remaining headroom is candidate-pool recall on inferential and multi-hop questions, whose answers are less lexically anchored to a single chunk.

> **Excluded runs.** Several earlier runs predate the methodology fixes (`n=20` samples, no persisted gold QA set, and `Recall@10` mislabelled, when it was actually Hit@3). These are **pre-methodology-fix** and are not comparable to the table above, so they are excluded rather than shown with misleading labels.

### Ablation

Component ablation over the fixed QA set, toggling `use_hybrid`, `use_reranker`, and `use_query_expansion`. Reported in its own table because the configs are not iterative improvements; they isolate each component's contribution. Re-run against the verified-clean index with claim-decomposition faithfulness, GPU-reranked, so retrieval latencies here are representative.

| Config | Recall@10 | MRR | Hit@3 | Faithfulness | Abstention | Retrieval p50 |
|---|---|---|---|---|---|---|
| baseline (dense only) | 0.92 | 0.862 | 0.88 | 0.924 | 0.16 | 0.012s |
| + hybrid (BM25) | 0.97 | 0.884 | 0.93 | 0.940 | 0.13 | 0.846s |
| + reranker | 0.92 | 0.862 | 0.92 | 0.907 | 0.10 | 0.247s |
| + expansion | 0.93 | 0.861 | 0.88 | 0.921 | 0.18 | 5.334s |
| **full** | **0.97** | **0.896** | **1.00** | **0.919** | **0.08** | **8.795s** |

Reading the ablation:

- **Hybrid is the single biggest contributor**: +5pp Recall@10 and +5pp Hit@3 over dense-only. BM25 recovers the exact-match queries dense search misses. The margins are smaller than on the polluted index — a clean candidate pool lifts the dense-only baseline most of all.
- **The reranker is a pure ordering win**: +4pp Hit@3 with an identical candidate pool (recall unchanged), and it nearly halves the cost of the full config's quality — only the combination of all three reaches Hit@3 = 1.00.
- **Query expansion is quality-neutral in isolation** (Hit@3 unchanged, +1pp recall) and only pays off when the reranker is there to filter the widened pool: full config gains +7pp Hit@3 over hybrid alone.
- **The expansion LLM call dominates retrieval latency** (5.33s p50 on its own), which is the cost of the full config's 8.80s p50.
- **Claim-level faithfulness is flat across configs** (0.91–0.94, no monotonic trend): grounding quality is set by the generator, not the retriever, once a plausible chunk is in context. Where retrieval components do show up is the abstention column — the full config abstains least (0.08) because better candidates clear the gate.

### Unanswerable / distractor evaluation

A separate `n=100` set of out-of-corpus questions, generated by streaming Wikipedia past the indexed 10,000 articles. This measures whether the pipeline correctly abstains instead of fabricating answers for questions the corpus cannot support.

| Metric | Value |
|---|---|
| Abstention accuracy | 0.99 |
| False answer rate | 0.01 |

On the verified-clean index (previous run: 0.96 / 0.04), the failure mode remains **semantic false positives**: the single false answer retrieved a plausible same-domain chunk from the *wrong* article (a question about the district of Günyüzü answered with a different town's date), which passes the gate and gets answered. This run was CPU-reranked; a GPU run may differ marginally.

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

This fetches the Wikipedia corpus, chunks, embeds, and persists the ChromaDB index to `.chroma/`. The corpus hash is combined with a config fingerprint, so changing the chunking or embedding config triggers a rebuild automatically. Pass `--force` to reindex unconditionally. Each build recreates the Chroma collection from scratch (a plain upsert would leave stale chunks behind when a config change shrinks the chunk count) and logs the final collection size alongside the upserted count.

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
python -m main serve --port 8000              # binds 127.0.0.1 by default
python -m main serve --host 0.0.0.0           # expose on the LAN (no auth — trusted networks only)
```

The `/query` endpoint validates its input: `question` is capped at 4000 chars, an optional `model` override must appear in `config.API_ALLOWED_MODELS`, and `filters` may only reference known scalar metadata keys. Internal error details are logged server-side, never returned to clients, and source metadata is stripped of local filesystem paths.

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
| `GenerationConfig` | `model`, `max_tokens`, `temperature`, `max_retries` | `qwen2.5:14b`, 1024, 0.0, 10 |

### Chunking

The semantic chunker splits documents at sentence-embedding distance peaks. `semantic_split_std_multiplier=1.5` splits at the top tail of sentence-boundary distances per document, producing 234,586 chunks across the 10k corpus — the best Hit@3 of the swept values (1.5, 2.5, 3.0, 3.5; everything above 2.5 plateaued with diminishing returns; sweep-era chunk counts were inflated by the since-fixed `chunk_recursive` duplication bug). `chunk_size_tokens=512` is the bge-large token limit and bounds chunk length; `chunk_size=1500` (chars) is retained only as a hard-split fallback. Because bge-large is asymmetric, the query-instruction prefix is applied to queries but not passages.

### Reranker score gating

If the top reranker score falls below `reranker_score_threshold`, the pipeline short-circuits to an abstention response before generation, rather than answering from weak context. The default threshold is `0.0` (effectively off until tuned from the observed score distribution), but the gate is what drives the unanswerable-set abstention behaviour once raised.

## Retrieval Design

Naive dense-only RAG fails in predictable ways, and each pipeline stage targets one:

**1. Vocabulary mismatch.** An exact-keyword query ("Battle of Hastings 1066") can score poorly against dense neighbours if the corpus phrases it differently. BM25 handles exact-term overlap natively. Dense and BM25 lists (one pair per query variant) are merged with **Reciprocal Rank Fusion**, which is rank-based, so it sidesteps the score-scale incompatibility between cosine similarity and BM25 scores. The ablation confirms this is the single largest quality lever.

**2. Phrasing brittleness.** A single phrasing of a question only probes one region of embedding space. **Query expansion** asks the LLM for two paraphrases; all three variants are embedded in one batched call and retrieved independently before fusion. In isolation it is quality-neutral, but combined with the reranker it lifts Hit@3, most visibly on paraphrased and multi-hop questions.

**3. Similarity ≠ relevance.** Top-*k* by cosine distance is not top-*k* by answer quality. A **cross-encoder reranker** scores each query–chunk pair jointly, attending to fine-grained interactions. It's expensive, so it runs only over the RRF-fused candidates and selects `rerank_top_n=3`. Selected chunks are neighbour-stitched into strict `[prev, current, next]` order (deduplicated) to restore local context for generation.

**4. Unsupported questions.** When the best candidate is still weak, answering is the wrong move. The **reranker score gate** abstains before generation. On the out-of-corpus set this yields 99% abstention accuracy at a 1% false-answer rate.

## Known Limitations

- **BM25 index is not persisted**: it is rebuilt in-memory from the chunk corpus on load. Cached per-process via `lru_cache`, but cold on every restart.
- **Chunk deduplication is disabled at scale**: the O(n²) exact-dedup approach is unusable above ~100k chunks. A FAISS approximate-nearest-neighbour rewrite is tracked separately.
- **Semantic false positives on statistics queries**: the remaining unanswerable-set failure is that entity-specific questions retrieve a plausible same-domain chunk from the wrong article and answer with the wrong figure. Tighter gating or entity-aware filtering is the likely fix.
- **Chunking mixes character and token units**: `chunk_recursive`'s hard-split fallback and overlap merge operate in characters while size limits are in tokens, so a small fraction of chunks (483 of 234,586, 0.2%) exceed the 512-token embedding limit and lose their tail in the dense vector. Stored text, BM25, and generation are unaffected.
