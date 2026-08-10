import pytest

pytest.importorskip("chromadb")

from tests.helpers import fake_embedding
from vectorstores.chroma import ChromaBackend


def test_upsert_and_search_ranks_closest_first():
    backend = ChromaBackend.in_memory()
    items = [
        {"id": 1, "content": "alpha", "source_path": "a.md", "chunk_index": 0, "embedding": fake_embedding("alpha")},
        {"id": 2, "content": "beta", "source_path": "b.md", "chunk_index": 1, "embedding": fake_embedding("beta")},
        {"id": 3, "content": "gamma", "source_path": "c.md", "chunk_index": 2, "embedding": fake_embedding("gamma")},
    ]
    backend.upsert(items)
    res = backend.search(fake_embedding("beta"), top_k=3)
    assert res, "expected results"
    assert res[0]["source_path"] == "b.md"  # query vector == beta's vector -> ranks first
    assert set(res[0]) == {"id", "content", "source_path", "chunk_index", "score"}
    assert -1e-6 <= res[0]["score"] <= 1.0 + 1e-6


def test_search_result_count_capped_by_top_k():
    backend = ChromaBackend.in_memory()
    backend.upsert(
        [{"id": i, "content": f"c{i}", "source_path": "s.md", "chunk_index": i, "embedding": fake_embedding(f"c{i}")} for i in range(5)]
    )
    assert len(backend.search(fake_embedding("c0"), top_k=2)) == 2
