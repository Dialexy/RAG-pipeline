from dataclasses import dataclass, field
from pathlib import Path


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CHROMA_PERSIST_DIR = Path(".chroma")


@dataclass
class ChunkConfig:
    strategy: str = "semantic"  # "fixed", "recursive", "semantic"
    chunk_size: int = 1500
    chunk_size_tokens: int = 512
    chunk_overlap: int = 64
    semantic_split_std_multiplier: float = 1.5
    semantic_min_chunk_size: int = 100


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 64
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrievalConfig:
    top_k: int = 10
    rerank_top_n: int = 3
    use_reranker: bool = True
    use_hybrid: bool = True  # BM25 + dense
    use_query_expansion: bool = True
    default_filters: dict | None = None
    reranker_score_threshold: float = 0.0


@dataclass
class GenerationConfig:
    model: str = "qwen2.5:14b"
    max_tokens: int = 1024
    temperature: float = 0.0
    # Ollama retry budget; long evals want persistence, the API wants to fail fast
    max_retries: int = 10


# Models the HTTP API may load via the request "model" field.
# The CLI --model flag is not restricted by this.
API_ALLOWED_MODELS: frozenset[str] = frozenset({"qwen2.5:14b"})


@dataclass
class PipelineConfig:
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
