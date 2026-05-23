# RAG Pipeline

A from-scratch Retrieval-Augmented Generation pipeline over a Wikipedia corpus. Built to study where naive RAG breaks and what targeted fixes look like: dense search alone undershoots on recall and conflates similarity with relevance; this pipeline adds semantic chunking, BM25 fusion, and cross-encoder reranking to address both.

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
  Chunking               semantic split (std_multiplier=2.0, chunk_size=1500)
        │                → ~63 chunks/article, recursive fallback for long segments
        ▼
  Embedding              SentenceTransformer → 384-dim vectors
        │
        ▼
  Vector Store           ChromaDB (persistent, batched upserts)


QUERY TIME
──────────────────────────────────────────────────────
User query
        │
        ├──────────────────────┐
        ▼                      ▼
  Dense Search           BM25 Search
  (ChromaDB ANN)         (rank-bm25, built in-memory)
        │                      │
        └──────────┬───────────┘
                   ▼
            RRF Fusion             top_k=10 merged candidates
                   │
                   ▼
          Cross-Encoder Rerank    BAAI/bge-reranker-large, top_n=3
                   │
                   ▼
            Generation            Ollama (qwen2.5:32b, local)
                   │
                   ▼
             Answer + Sources
```

## Tech Stack

| Component | Library / Model |
|---|---|
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` |
| Vector store | `chromadb` (persistent) |
| Sparse retrieval | `rank-bm25` |
| Reranker | `BAAI/bge-reranker-large` |
| Generation | Ollama, `qwen2.5:32b-instruct-q4_K_M` (local) |
| Corpus | Hugging Face `datasets`, Wikipedia |

## Benchmark Results

Evaluated on 20 QA pairs generated from the 10,000-article Wikipedia corpus. Each run reflects incremental improvements to the pipeline.

| Run | Model | Articles | Recall@10 | MRR | Faithfulness |
|---|---|---|---|---|---|
| Baseline (recursive chunking) | 14B | 10k | 0.90 | 0.90 | 0.60 |
| Semantic chunking | 14B | 10k | 0.90 | 0.90 | 0.76 |
| 32B judge (pre-improvements) | 32B | 10k | 0.95 | 0.95 | 0.74 |
| chunk_size=1500, all eval fixes | 32B | 10k | 1.00 | 1.00 | 0.715 |
| **GPU reranker (BAAI/bge-reranker-large)** | **32B** | **10k** | **1.00** | **0.975** | **0.875** |

**Recall@10** and **MRR** measure retrieval quality. Both near 1.0 indicate the hybrid retriever consistently finds and top-ranks the relevant context.

**Faithfulness** (0.875) measures generation grounding: whether the answer stays within what the retrieved context supports. The jump from 0.715 to 0.875 came from upgrading the reranker to `BAAI/bge-reranker-large` on GPU, which surfaces more precisely relevant chunks before generation.

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally with enough VRAM for your chosen model. On a 12 GiB card, set `OLLAMA_GPU_OVERHEAD=3221225472` in the Ollama service environment to reserve 3 GiB for the reranker and embedding model.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pull the generation model:

```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```

## Usage

**Index**: run once (or whenever the corpus changes):

```bash
python -m main index
```

This fetches the Wikipedia corpus, chunks, embeds, and persists the ChromaDB index to `.chroma/`. Pass `--force` to reindex even if the corpus hash is unchanged.

**Query:**

```bash
python -m main query "What is the speed of light?"
python -m main query "Who invented the printing press?"
```

Output is the generated answer printed to stdout. Sources with reranker scores are available via the Python API:

```python
from config import PipelineConfig
from src.pipeline import build_pipeline, query_pipeline

cfg = PipelineConfig()

# Index time: run once
build_pipeline(cfg)

# Query time
result = query_pipeline("What is the speed of light?", cfg)
print(result["answer"])
for src in result["sources"]:
    print(src["metadata"]["fileName"], src.get("reranked_score"))
```

**Evaluate:**

```bash
python -m eval.evaluate
```

Runs 20 synthetic QA pairs and reports Recall@10, MRR, and faithfulness.

## Project Structure

```
main.py                CLI entry point (index / query subcommands)
config.py              Dataclass-based config for all pipeline stages
requirements.txt       Pinned dependencies
src/
  models.py            Document dataclass
  ingestion.py         Corpus download and document iteration (Hugging Face datasets)
  chunking.py          Semantic, recursive, and fixed-size chunking strategies
  embedding.py         SentenceTransformer batched encoding
  vector_store.py      ChromaDB index build and ANN search
  retrieval.py         BM25, RRF fusion, cross-encoder reranking
  generation.py        Ollama prompt construction and inference
  pipeline.py          build_pipeline() and query_pipeline() orchestration
eval/
  evaluate.py          Recall@10, MRR, and faithfulness evaluation loop
data/
  raw/                 Downloaded .txt documents
.chroma/               Persistent ChromaDB vector index
```

## Configuration

All knobs live in `config.py` as frozen dataclasses:

| Config | Key fields | Defaults |
|---|---|---|
| `ChunkConfig` | `strategy`, `chunk_size`, `chunk_overlap`, `semantic_split_std_multiplier` | `semantic`, 1500, 64, 2.0 |
| `EmbeddingConfig` | `model_name`, `batch_size` | `all-MiniLM-L6-v2`, 64 |
| `RetrievalConfig` | `top_k`, `rerank_top_n`, `use_hybrid`, `use_reranker` | 10, 3, True, True |
| `GenerationConfig` | `model`, `max_tokens`, `temperature` | `qwen2.5:32b-instruct-q4_K_M`, 1024, 0.0 |

### Chunking

The semantic chunker splits documents at sentence-embedding distance peaks. `semantic_split_std_multiplier=2.0` triggers a split only at the top ~2.3% of sentence-boundary distances per document, producing ~2 semantic chunks per article on average. Any semantic chunk exceeding `chunk_size` is further split by the recursive fallback. At `chunk_size=1500` this yields ~63 chunks per article across the 10k Wikipedia corpus.

Near-duplicate deduplication (`deduplicate_chunks`) is implemented but disabled at scale — the O(n²) approach is unusable above ~100k chunks. A FAISS-based approximate rewrite is planned.

## Retrieval Design

Naive dense-only RAG fails in two predictable ways:

**1. Vocabulary mismatch.** A query containing exact keywords ("Battle of Hastings 1066") may score poorly against dense embedding neighbors if the corpus uses different phrasing. BM25 handles exact-term overlap natively. The two retrievers are fused with **Reciprocal Rank Fusion (RRF)**, which is rank-based (not score-based) so it avoids the score-scale incompatibility between cosine similarity and BM25 scores.

**2. Similarity ≠ relevance.** The top-*k* by cosine distance is not the top-*k* by answer quality; embeddings compress semantics globally, not in relation to a specific question. A **cross-encoder reranker** (`BAAI/bge-reranker-large`) scores each query–chunk pair jointly by feeding both as a single input, attending to fine-grained interactions. It's expensive, so it runs over the RRF-fused top-10 and selects `rerank_top_n=3` for generation.

The three-stage funnel (dense+BM25 → RRF → cross-encoder) trades latency for precision: the cheap retrievers widen the candidate set, and the expensive reranker refines it.

## Known Limitations

- **BM25 index is not persisted**: rebuilt in-memory on every query from the ChromaDB corpus.
- **BM25 granularity mismatch**: BM25 currently searches over chunks; dense search also searches chunks — but BM25 relevance scoring would benefit from document-level context. Fix planned.
- **Chunk deduplication disabled**: O(n²) exact dedup is too slow at 600k+ chunks. FAISS-based approximate rewrite is planned.
- **Embedding model**: `all-MiniLM-L6-v2` is fast but undersized. Upgrade to `BAAI/bge-large-en-v1.5` planned.

## Planned

- **Embedding model upgrade**: swap `all-MiniLM-L6-v2` for `BAAI/bge-large-en-v1.5`, re-index at 10k.
- **Chunk deduplication v2**: FAISS approximate nearest-neighbour dedup.
- **Separate evaluator model**: use 32B as judge only; Qwen 14B for generation to reduce eval cost.
- **Claim decomposition faithfulness**: split answers into atomic claims, evaluate each independently.
- **BM25 granularity fix**: align BM25 to chunk-level corpus to match dense search granularity.
- **`python -m main serve`**: start uvicorn programmatically as a subcommand.
- **Test suite**: end-to-end and unit tests once pipeline is stable.
