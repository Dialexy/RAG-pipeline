# RAG Pipeline

A from-scratch Retrieval-Augmented Generation pipeline over a Wikipedia corpus. Built to study where naive RAG breaks and what targeted fixes look like: dense search alone undershoots on recall and conflates similarity with relevance; this pipeline adds BM25 fusion and cross-encoder reranking to address both.

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
  Chunking               recursive character split (512 tok, 64 overlap)
        │
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
          Cross-Encoder Rerank    ms-marco-MiniLM-L-6-v2, top_n=3
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
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` |
| Vector store | `chromadb` (persistent) |
| Sparse retrieval | `rank-bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama, `qwen2.5:14b` (local) |
| Corpus | Hugging Face `datasets`, Wikipedia |

## Evaluation

Evaluated on 20 QA pairs generated from the 10,000-article Wikipedia corpus.

| Metric | Score |
|---|---|
| Recall@10 | **0.90** |
| MRR | **0.90** |
| Faithfulness | **0.60** |

**Recall@10** and **MRR** measure retrieval: whether the relevant chunk appears in the top-10 results, and how highly it ranks. Both at 0.9 indicate the hybrid retriever is finding the right context reliably.

**Faithfulness** (0.60) measures generation: whether the answer is grounded in the retrieved context rather than hallucinated. The gap between retrieval and generation quality points at the model occasionally over-generating beyond what the context supports, a known failure mode for smaller local models.

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pull the generation model:

```bash
ollama pull qwen2.5:14b
```

## Usage

**Index**: run once (or whenever the corpus changes):

```bash
python -m main index
```

This fetches the Wikipedia corpus, chunks, embeds, and persists the ChromaDB index to `.chroma/`.

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

## Project Structure

```
main.py                CLI entry point (index / query subcommands)
config.py              Dataclass-based config for all pipeline stages
requirements.txt       Pinned dependencies
src/
  models.py            Document dataclass
  ingestion.py         Corpus download and document iteration (Hugging Face datasets)
  chunking.py          Fixed-size and recursive chunking strategies
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
| `ChunkConfig` | `strategy`, `chunk_size`, `chunk_overlap` | `recursive`, 512, 64 |
| `EmbeddingConfig` | `model_name`, `batch_size` | `all-MiniLM-L6-v2`, 64 |
| `RetrievalConfig` | `top_k`, `rerank_top_n`, `use_hybrid`, `use_reranker` | 10, 3, True, True |
| `GenerationConfig` | `model`, `max_tokens`, `temperature` | `qwen2.5:14b`, 1024, 0.0 |

## Retrieval Design

Naive dense-only RAG fails in two predictable ways:

**1. Vocabulary mismatch.** A query containing exact keywords ("Battle of Hastings 1066") may score poorly against dense embedding neighbors if the corpus uses different phrasing. BM25 handles exact-term overlap natively. The two retrievers are fused with **Reciprocal Rank Fusion (RRF)**, which is rank-based (not score-based) so it avoids the score-scale incompatibility between cosine similarity and BM25 scores.

**2. Similarity ≠ relevance.** The top-*k* by cosine distance is not the top-*k* by answer quality; embeddings compress semantics globally, not in relation to a specific question. A **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) scores each query–chunk pair jointly by feeding both as a single input, attending to fine-grained interactions. It's expensive, so it runs over the RRF-fused top-10 and selects `rerank_top_n=3` for generation.

The three-stage funnel (dense+BM25 → RRF → cross-encoder) trades latency for precision: the cheap retrievers widen the candidate set, and the expensive reranker refines it.

## Known Limitations

- **BM25 index is not persisted**: rebuilt in-memory on every query from the ChromaDB corpus.
- **Reranker model is re-instantiated per call**: no caching between queries.
- **Faithfulness ceiling**: at 0.60, the generation model (14B, local) still over-generates. A larger model or stricter prompt constraints would help.
- **Semantic chunking**: `chunk_semantic()` is stubbed but not implemented; the pipeline uses recursive splitting only.

## Planned

- **Faithfulness improvement**: tighter system prompt and citation-grounding instructions to reduce over-generation.
- **Semantic chunking**: implement the stubbed `chunk_semantic()` using embedding-based boundary detection.
- **FastAPI wrapper**: expose `query_pipeline` as a REST endpoint for easier integration.
- **`--model` CLI flag**: override `GenerationConfig.model` at runtime without editing `config.py`.
