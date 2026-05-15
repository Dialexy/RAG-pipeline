# RAG Pipeline

A from-scratch Retrieval-Augmented Generation pipeline over a Wikipedia corpus. Built to study where naive RAG breaks and what targeted fixes look like.

## Architecture

```
Corpus (Wikipedia)
      │
      ▼
  Ingestion          fetch_corpus() → data/raw/*.txt
      │
      ▼
  Chunking           fixed / recursive character split
      │
      ▼
  Embedding          SentenceTransformer (all-MiniLM-L6-v2)
      │
      ▼
  Vector Store       ChromaDB (persistent, batched upserts)
      │
      ▼
  Retrieval          dense search + BM25 → RRF fusion → cross-encoder rerank
      │
      ▼
  Generation         Ollama (qwen2.5:3b, local)
```

## Project Structure

```
config.py              dataclass-based config (chunking, embedding, retrieval, generation)
src/
  models.py            Document dataclass
  ingestion.py         corpus download and document iteration
  chunking.py          fixed-size and recursive chunking strategies
  embedding.py         SentenceTransformer batched encoding
  vector_store.py      ChromaDB index build and dense search
  retrieval.py         BM25, RRF, cross-encoder reranking
  generation.py        Ollama prompt construction and inference
  pipeline.py          build_pipeline() and query_pipeline() entry points
data/
  raw/                 downloaded .txt documents
```

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally with `qwen2.5:3b` pulled.

```bash
python -m venv .venv && source .venv/bin/activate
pip install sentence-transformers chromadb rank-bm25 datasets ollama
```

Pull the generation model:

```bash
ollama pull qwen2.5:3b
```

## Usage

```python
from config import PipelineConfig
from src.pipeline import build_pipeline, query_pipeline

cfg = PipelineConfig()

# Index time — run once or when corpus changes
build_pipeline(cfg)

# Query time
result = query_pipeline("What is the speed of light?", cfg)
print(result["answer"])
for src in result["sources"]:
    print(src["metadata"]["fileName"], src.get("reranked_score"))
```

## Configuration

All knobs live in `config.py`:

| Config | Key fields | Defaults |
|---|---|---|
| `ChunkConfig` | `strategy`, `chunk_size`, `chunk_overlap` | `recursive`, 512, 64 |
| `EmbeddingConfig` | `model_name`, `batch_size` | `all-MiniLM-L6-v2`, 64 |
| `RetrievalConfig` | `top_k`, `rerank_top_n`, `use_hybrid`, `use_reranker` | 10, 3, True, True |
| `GenerationConfig` | `model`, `max_tokens`, `temperature` | `qwen2.5:3b`, 1024, 0.0 |

## Retrieval Design

Naive dense-only RAG fails in two predictable ways:

1. **Vocabulary mismatch** — a query with exact keywords scores poorly against embedding-space neighbors. Fix: add BM25 as a second retriever and fuse with Reciprocal Rank Fusion.
2. **Similarity ≠ relevance** — the top-*k* by cosine distance is not the top-*k* by answer quality. Fix: cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) scores query–chunk pairs jointly and selects `rerank_top_n` << `top_k`.

## What's Missing

- [ ] **Evaluation** — no RAGAS / BEIR metrics yet; retrieval quality is qualitative only
- [ ] **Semantic chunking** — `chunk_semantic()` is stubbed but not implemented
- [ ] **BM25 index persistence** — rebuilt from scratch on every query
- [ ] **Reranker caching** — CrossEncoder is re-instantiated on every `rerank()` call
- [ ] **Corpus scale** — capped at 100 Wikipedia articles for development
