"""
Chroma wrapper: index chunks and run similarity search.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone

import chromadb
from config import PipelineConfig, CHROMA_PERSIST_DIR
from .models import Document
from .embedding import embed_chunks
from chromadb.api.models.Collection import Collection
from .logger import get_logger


logger = get_logger(__name__)


def build_index(chunks: list[Document], cfg: PipelineConfig) -> None:
    """
    Embed all chunks and upsert into Chroma.
    Persists to CHROMA_PERSIST_DIR.
    """
    logger.info("Building index for %d chunks", len(chunks))
    chroma_client = chromadb.PersistentClient(CHROMA_PERSIST_DIR)

    rag_chunks = chroma_client.get_or_create_collection(
        name="rag_chunks", metadata={"hnsw:space": "cosine"}
    )
    rag_chunks.modify(
        metadata={
            "config": json.dumps(
                {"chunking": asdict(cfg.chunking), "embedding": asdict(cfg.embedding)},
                sort_keys=True,
            ),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    texts = [doc.text for doc in chunks]
    embedded = embed_chunks(texts, cfg.embedding)

    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        rag_chunks.upsert(
            ids=[chunk.id for chunk in batch],
            embeddings=embedded[i : i + BATCH_SIZE].tolist(),
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
        )

    logger.info("Index build complete: %d chunks upserted into Chroma", len(chunks))


def load_index(cfg: PipelineConfig) -> Collection:
    """Return an existing Chroma collection from disk."""
    logger.info("Loading index from %s", CHROMA_PERSIST_DIR)
    chroma_client = chromadb.PersistentClient(CHROMA_PERSIST_DIR)
    try:
        rag_chunks = chroma_client.get_collection(name="rag_chunks")
    except Exception as e:
        raise RuntimeError("Index not found. Run 'python -m main index' first.") from e
    return rag_chunks


def load_chunk_corpus(collection: Collection) -> list[Document]:
    """Load all chunks from a Chroma collection as Document objects."""
    corpus: list[Document] = []
    batch_size = 5000
    offset = 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"], limit=batch_size, offset=offset
        )
        if not batch["ids"]:
            break
        for id_, text, meta in zip(
            batch["ids"], batch["documents"] or [], batch["metadatas"] or []
        ):
            corpus.append(Document(id=id_, text=text, metadata=dict(meta)))
        offset += batch_size
    logger.info("Loaded %d chunks from collection", len(corpus))
    return corpus


def fetch_neighbouring_chunks(
    chunk_id: str, collection
) -> tuple[str | None, str | None]:
    """Given a chunk id like '90.txt::chunk7', return (prev_text, next_text), either may be None."""

    parts = chunk_id.split("::chunk")
    doc_id = parts[0]
    chunk_index = int(parts[1])

    def _fetch(cid: str) -> str | None:
        try:
            result = collection.get(ids=[cid], include=["documents"])
            if result["documents"] and result["documents"][0]:
                return result["documents"][0]
        except Exception as e:
            logger.warning("Neighbour lookup failed for %s: %s", cid, e)
        return None

    prev_text = _fetch(f"{doc_id}::chunk{chunk_index - 1}")
    next_text = _fetch(f"{doc_id}::chunk{chunk_index + 1}")

    logger.debug(
        "Neighbours for %s: prev=%s next=%s",
        chunk_id,
        "found" if prev_text else "none",
        "found" if next_text else "none",
    )
    return prev_text, next_text


def dense_search(
    query_embedding, top_k: int, collection, filters: dict | None = None
) -> list[dict]:
    """Return top-k chunks by cosine distance (score is distance: lower is better)."""
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=filters,
    )

    return [
        {
            "text": doc,
            "metadata": meta,
            "score": float(dist),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
