"""
Chroma wrapper: index chunks and run similarity search.
"""

from config import PipelineConfig, CHROMA_PERSIST_DIR
from .models import Document
from .embedding import embed_chunks
import chromadb
from chromadb.api.models.Collection import Collection


def build_index(chunks: list[Document], cfg: PipelineConfig) -> None:
    """
    Embed all chunks and upsert into Chroma.
    Persists to CHROMA_PERSIST_DIR.
    """
    chroma_client = chromadb.PersistentClient(CHROMA_PERSIST_DIR)

    rag_chunks = chroma_client.get_or_create_collection(name="rag_chunks")
    texts = [doc.text for doc in chunks]
    embedded = embed_chunks(texts, cfg.embedding)

    rag_chunks.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embedded.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )


def load_index(cfg: PipelineConfig) -> Collection:
    """Return an existing Chroma collection from disk."""
    chroma_client = chromadb.PersistentClient(CHROMA_PERSIST_DIR)
    rag_chunks = chroma_client.get_collection(name="rag_chunks")
    return rag_chunks


def dense_search(query_embedding, top_k: int, collection) -> list[dict]:
    """Return top-k chunks by cosine similarity."""
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": doc,
            "metadata": meta,
            "score": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
