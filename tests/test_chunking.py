from src.chunking import chunk_document
from src.models import Document
from config import ChunkConfig, EmbeddingConfig

LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 200

DOC = Document(id="test-doc", text=LONG_TEXT, metadata={})
EMBEDDING_CFG = EmbeddingConfig()


# --- fixed strategy ---


def test_fixed_chunks_within_size():
    cfg = ChunkConfig(strategy="fixed", chunk_size=200, chunk_overlap=20)
    chunks = list(chunk_document(DOC, cfg, EMBEDDING_CFG))
    assert all(len(c.text) <= cfg.chunk_size for c in chunks)


def test_fixed_produces_multiple_chunks():
    cfg = ChunkConfig(strategy="fixed", chunk_size=200, chunk_overlap=20)
    chunks = list(chunk_document(DOC, cfg, EMBEDDING_CFG))
    assert len(chunks) > 1


# --- recursive strategy ---


def test_recursive_chunks_within_size():
    # chunk_size_tokens=20 forces splits; merged chunks = overlap + sentence ≪ 200 chars
    cfg = ChunkConfig(
        strategy="recursive", chunk_size=200, chunk_size_tokens=20, chunk_overlap=20
    )
    chunks = list(chunk_document(DOC, cfg, EMBEDDING_CFG))
    assert all(len(c.text) <= cfg.chunk_size for c in chunks)


def test_recursive_produces_multiple_chunks():
    cfg = ChunkConfig(
        strategy="recursive", chunk_size=200, chunk_size_tokens=20, chunk_overlap=20
    )
    chunks = list(chunk_document(DOC, cfg, EMBEDDING_CFG))
    assert len(chunks) > 1
