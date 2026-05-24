"""
FastAPI app exposing the RAG pipeline over HTTP.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.vector_store import load_index
from src.ingestion import iter_documents
from config import RAW_DIR, PipelineConfig
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
    state["corpus"] = list(iter_documents(RAW_DIR))
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
    cfg = state["cfg"]
    # TODO: I need to avoid mutating the shared cfg state. I need to Create a copy per request for thread safety
    if request.model is not None:
        cfg.generation.model = request.model
    result = query_pipeline(
        request.question,
        cfg,
        collection=state["collection"],
        corpus=state["corpus"],
        filters=request.filters,
    )
    return QueryResponse(
        answer=result["answer"], sources=result["sources"], query=result["query"]
    )
