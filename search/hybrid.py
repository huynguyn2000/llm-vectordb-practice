"""Hybrid retrieval: fuse vector and keyword search with RRF.

Runs vector (cosine) and keyword (tsvector) search independently, each
returning up to candidate_k ranked chunks, then merges their rankings with
reciprocal rank fusion and returns the top_k as ChunkResult objects. The
returned score is the RRF score.
"""

from core.db import VectorStore
from core.embedder import Embedder
from core.models import ChunkResult
from search.fusion import reciprocal_rank_fusion


def hybrid_search(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int = 3,
    candidate_k: int = 20,
) -> list[ChunkResult]:
    if not query.strip():
        return []

    embedding = embedder.embed(query)
    vec_rows = store.search_chunks(embedding, top_k=candidate_k)
    kw_rows = store.search_chunks_keyword(query, top_k=candidate_k)

    rows_by_id = {r["id"]: r for r in vec_rows + kw_rows}  # union, dedup by id
    fused = reciprocal_rank_fusion(
        [[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]]
    )
    return [
        ChunkResult(**{**rows_by_id[doc_id], "score": score})
        for doc_id, score in fused[:top_k]
    ]
