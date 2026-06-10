"""
FastAPI app exposing the RAG pipeline over HTTP.
"""

import dataclasses
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from src.vector_store import load_index, load_chunk_corpus
from config import API_ALLOWED_MODELS, PipelineConfig
from src.pipeline import query_pipeline
from src.logger import get_logger

logger = get_logger(__name__)

# Metadata keys clients may filter on; everything else is rejected.
FILTERABLE_KEYS = {
    "fileName",
    "extension",
    "chunk_strategy",
    "parent_id",
    "chunk_index",
}

# Metadata keys stripped from responses (local filesystem details).
PRIVATE_METADATA_KEYS = {"path", "last_modified"}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    filters: dict | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    error: str | None = None


state = {}


@asynccontextmanager
async def lifespan(app):
    """
    Load config, Chroma index, and document corpus on startup; clear state on shutdown.
    Shared via module-level `state` dict so route handlers don't re-initialise per request.
    """
    cfg = PipelineConfig()
    # Fail fast over HTTP: the default retry budget (10 attempts with sleeps
    # up to 90s) is meant for unattended evals and would block a request
    # handler for minutes.
    cfg = dataclasses.replace(
        cfg, generation=dataclasses.replace(cfg.generation, max_retries=2)
    )
    state["cfg"] = cfg
    state["collection"] = load_index(cfg)
    state["corpus"] = load_chunk_corpus(state["collection"])
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


def _validate_filters(filters: dict | None) -> None:
    if filters is None:
        return
    for key, value in filters.items():
        if key not in FILTERABLE_KEYS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown filter key {key!r}. Allowed: {sorted(FILTERABLE_KEYS)}",
            )
        if not isinstance(value, (str, int, float, bool)):
            raise HTTPException(
                status_code=422,
                detail=f"Filter values must be scalars, got {type(value).__name__} for {key!r}",
            )


def _public_sources(sources: list[dict]) -> list[dict]:
    return [
        {
            **source,
            "metadata": {
                k: v
                for k, v in source.get("metadata", {}).items()
                if k not in PRIVATE_METADATA_KEYS
            },
        }
        for source in sources
    ]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Embed the question, retrieve chunks, generate an answer, return structured response."""
    _validate_filters(request.filters)
    cfg = state["cfg"]
    if request.model is not None:
        if request.model not in API_ALLOWED_MODELS:
            raise HTTPException(
                status_code=422,
                detail=f"Model not available. Allowed: {sorted(API_ALLOWED_MODELS)}",
            )
        cfg = dataclasses.replace(
            cfg, generation=dataclasses.replace(cfg.generation, model=request.model)
        )
    try:
        result = query_pipeline(
            request.question,
            cfg,
            collection=state["collection"],
            corpus=state["corpus"],
            filters=request.filters,
        )
    except Exception:
        logger.exception("Query pipeline failed for question: %r", request.question)
        raise HTTPException(status_code=500, detail="Internal server error")

    # query_pipeline reports handled failures via "error"; log the detail
    # server-side and return a generic marker instead of internals.
    if result.get("error") is not None:
        logger.error(
            "Pipeline error for question %r: %s", request.question, result["error"]
        )
    return QueryResponse(
        answer=result["answer"],
        sources=_public_sources(result["sources"]),
        query=result["query"],
        error="internal error" if result.get("error") is not None else None,
    )
