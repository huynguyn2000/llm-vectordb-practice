import pytest

from search.hybrid import hybrid_search
from tests.helpers import FakeEmbedder, _purge_test_rows, fake_embedding

pytestmark = pytest.mark.integration

ROOT = "zz-test-hy"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def _insert(store, path, content):
    store.upsert_source_with_chunks(
        ROOT, path, "test-hash-" + path, [(content, 3, fake_embedding(content))]
    )


def test_returns_chunk_results(store):
    _insert(store, "zz-test-hy/a.md", "The frobnicator handles edge cases.")
    results = hybrid_search("frobnicator", store, FakeEmbedder(), top_k=3)
    assert results and all(hasattr(r, "source_path") for r in results)


def test_keyword_distinctive_term_is_retrieved(store):
    # A term FakeEmbedder's vector won't surface, but keyword search will.
    _insert(store, "zz-test-hy/kw.md", "Xylophone zebra quokka distinctive term.")
    results = hybrid_search("quokka", store, FakeEmbedder(), top_k=5)
    assert any(r.source_path == "zz-test-hy/kw.md" for r in results)


def test_chunk_in_both_engines_is_deduped(store):
    _insert(store, "zz-test-hy/dup.md", "photosynthesis photosynthesis photosynthesis")
    results = hybrid_search("photosynthesis", store, FakeEmbedder(), top_k=20)
    paths = [r.source_path for r in results]
    assert paths.count("zz-test-hy/dup.md") == 1


def test_empty_query_returns_empty(store):
    assert hybrid_search("   ", store, FakeEmbedder(), top_k=3) == []
