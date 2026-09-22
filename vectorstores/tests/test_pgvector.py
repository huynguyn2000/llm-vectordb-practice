import pytest

from tests.helpers import _purge_test_rows, fake_embedding
from vectorstores.pgvector import PgvectorBackend

pytestmark = pytest.mark.integration

ROOT = "zz-test-vdb"
PATH = "zz-test-vdb/a.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def test_pgvector_backend_search_shape(store):
    store.upsert_source_with_chunks(
        ROOT, PATH, "hash", [("alpha content", 3, fake_embedding("alpha content"))]
    )
    backend = PgvectorBackend(store)
    assert backend.name == "pgvector"
    res = backend.search(fake_embedding("alpha content"), top_k=5)
    ours = [r for r in res if r["source_path"] == PATH]
    assert ours
    assert set(ours[0]) == {"id", "content", "source_path", "chunk_index", "score"}
