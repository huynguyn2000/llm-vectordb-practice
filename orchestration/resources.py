"""Dagster resources wrapping the project's data dependencies, so assets stay
testable (swap EmbedderResource for a fake in tests). These construct clients
lazily in their getters — instantiating the resource opens no connections."""

from dagster import ConfigurableResource

from core.db import VectorStore
from core.embedder import Embedder


class VectorStoreResource(ConfigurableResource):
    """Provides a pgvector VectorStore. The caller is responsible for closing it."""

    def get_store(self) -> VectorStore:
        return VectorStore()


class EmbedderResource(ConfigurableResource):
    """Provides an Ollama Embedder."""

    def get_embedder(self) -> Embedder:
        return Embedder()
