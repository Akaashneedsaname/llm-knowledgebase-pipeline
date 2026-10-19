"""
Pluggable embedding providers.

- local:  fastembed (ONNX, CPU-only, no API key) with BAAI/bge-small-en-v1.5, 384-dim
- gemini: Gemini embedding API, 768-dim, with retrieval-specific task types
"""

import math
from functools import lru_cache

from kb.config import settings


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class LocalEmbedder:
    """Offline embeddings with fastembed."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)
        self.dim = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class GeminiEmbedder:
    """Cloud embeddings with the Gemini API."""

    def __init__(self, model_name: str, api_key: str, dim: int = 768, batch_size: int = 100):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts[i:i + self.batch_size],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            )
            # Truncated Gemini embeddings must be re-normalized for cosine similarity
            vectors.extend(_normalize(e.values) for e in result.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


@lru_cache(maxsize=1)
def get_embedder():
    """Return the embedder for the configured provider (cached)."""
    if settings.embedding_provider == "gemini":
        return GeminiEmbedder(settings.gemini_embedding_model, settings.gemini_api_key, settings.embedding_dim)
    return LocalEmbedder(settings.local_embedding_model)
