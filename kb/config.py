"""
Centralized configuration, loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # PostgreSQL + pgvector
    database_url: str = os.getenv("DATABASE_URL", "postgresql://kb:kb@localhost:5433/kb")

    # Embeddings: "local" (fastembed, runs offline) or "gemini" (cloud)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # LLM (optional): used for document labeling and answer generation
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_llm_model: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

    # OCR
    ocr_min_chars: int = int(os.getenv("OCR_MIN_CHARS", "40"))  # below this, a PDF page is treated as scanned
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))
    ocr_lang: str = os.getenv("OCR_LANG", "eng")

    # Chunking (tokens)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))

    @property
    def embedding_dim(self) -> int:
        """Vector size for the active embedding provider."""
        return 768 if self.embedding_provider == "gemini" else 384

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    def validate(self) -> list[str]:
        """Return a list of configuration problems."""
        problems = []
        if self.embedding_provider not in ("local", "gemini"):
            problems.append("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        if self.chunk_overlap >= self.chunk_size:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return problems


# Global settings instance
settings = Settings()
