"""Environment-backed settings for the RAG service."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", "data/qdrant"))
    collection: str = os.getenv("QDRANT_COLLECTION", "documents")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    retrieval_candidates: int = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))


settings = Settings()
