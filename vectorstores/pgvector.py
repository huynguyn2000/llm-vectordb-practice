"""pgvector-backed VectorBackend (wraps the existing VectorStore)."""

from core.db import VectorStore


class PgvectorBackend:
    name = "pgvector"

    def __init__(self, store: VectorStore):
        self._store = store

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        return self._store.search_chunks(embedding, top_k=top_k)
