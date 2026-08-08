import pytest

from tests.helpers import _purge_test_rows, fake_embedding

pytestmark = pytest.mark.integration

ROOT = "zz-test-kw"
PATH = "zz-test-kw/doc.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def _insert(store, content):
    store.upsert_source_with_chunks(ROOT, PATH, "test-hash", [(content, 3, fake_embedding(content))])


def test_keyword_finds_exact_term(store):
    _insert(store, "Photosynthesis converts sunlight into glucose.")
    rows = store.search_chunks_keyword("photosynthesis", top_k=20)
    assert any(r["source_path"] == PATH for r in rows)


def test_keyword_no_match_returns_empty(store):
    _insert(store, "Photosynthesis converts sunlight into glucose.")
    rows = store.search_chunks_keyword("zqxjkbrstvwx", top_k=20)
    assert rows == []


def test_keyword_garbage_input_does_not_raise(store):
    # websearch_to_tsquery must swallow operator garbage instead of erroring.
    assert store.search_chunks_keyword('"unterminated AND OR -', top_k=20) == []


def test_keyword_result_shape_matches_vector(store):
    _insert(store, "Photosynthesis converts sunlight into glucose.")
    rows = store.search_chunks_keyword("photosynthesis", top_k=20)
    ours = [r for r in rows if r["source_path"] == PATH]
    assert set(ours[0]) == {"id", "content", "chunk_index", "source_path", "score"}
