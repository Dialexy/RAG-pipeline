"""
FastAPI app exposing the RAG pipeline over HTTP.
"""

import dataclasses
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.vector_store import load_index, load_chunk_corpus
from config import PipelineConfig
from src.pipeline import query_pipeline


class QueryRequest(BaseModel):
    question: str
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


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Embed the question, retrieve chunks, generate an answer, return structured response."""
    try:
        cfg = state["cfg"]
        if request.model is not None:
            cfg = dataclasses.replace(
                cfg, generation=dataclasses.replace(cfg.generation, model=request.model)
            )
        result = query_pipeline(
            request.question,
            cfg,
            collection=state["collection"],
            corpus=state["corpus"],
            filters=request.filters,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"],
            error=result.get("error"),
        )
    except Exception as e:
        return QueryResponse(
            answer="",
            sources=[],
            query=request.question,
            error=str(e),
        )
