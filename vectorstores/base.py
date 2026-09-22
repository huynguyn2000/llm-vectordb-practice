"""Vector backend protocol — the search interface both stores share."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    name: str

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return up to top_k results, each a dict with keys id, content,
        source_path, chunk_index, score (cosine similarity, higher = better)."""
        ...
