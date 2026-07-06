import pytest

from tests.helpers import fake_embedding

pytestmark = pytest.mark.integration

PATH_A = "zz-test-db/a.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    store.delete_source(PATH_A)
    yield
    store.delete_source(PATH_A)


def _rows(texts):
    return [(t, 3, fake_embedding(t)) for t in texts]


def test_upsert_and_hash_roundtrip(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha", "beta"]))
    assert store.get_source_hashes()[PATH_A] == "hash1"


def test_upsert_replaces_previous_chunks(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha", "beta"]))
    store.upsert_source_with_chunks(PATH_A, "hash2", _rows(["gamma"]))
    assert store.get_source_hashes()[PATH_A] == "hash2"
    ours = [
        r
        for r in store.search_chunks(fake_embedding("gamma"), top_k=50)
        if r["source_path"] == PATH_A
    ]
    assert [r["content"] for r in ours] == ["gamma"]


def test_delete_source_cascades_to_chunks(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha"]))
    store.delete_source(PATH_A)
    assert PATH_A not in store.get_source_hashes()
    results = store.search_chunks(fake_embedding("alpha"), top_k=50)
    assert all(r["source_path"] != PATH_A for r in results)


def test_search_chunks_result_shape(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha"]))
    ours = [
        r
        for r in store.search_chunks(fake_embedding("alpha"), top_k=50)
        if r["source_path"] == PATH_A
    ]
    assert set(ours[0]) == {"id", "content", "chunk_index", "source_path", "score"}
