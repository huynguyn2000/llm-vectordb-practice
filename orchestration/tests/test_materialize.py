"""Materialize the assets + check against real Postgres with a FakeEmbedder
resource (no Ollama). CORPUS_DIR is pointed at a temp dir of zz-test files so
the real corpus's embeddings are never overwritten."""

import pytest
from dagster import ConfigurableResource, materialize

from orchestration.assets import chunks_present, corpus_source, pgvector_chunks
from orchestration.resources import VectorStoreResource
from tests.helpers import FakeEmbedder, _purge_test_rows

pytestmark = pytest.mark.integration


class FakeEmbedderResource(ConfigurableResource):
    def get_embedder(self):
        return FakeEmbedder()


@pytest.fixture
def temp_corpus(tmp_path, monkeypatch):
    (tmp_path / "zz-test-doc.md").write_text(
        "# Test\n\nDagster orchestration integration test document.",
        encoding="utf-8",
    )
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def test_materialize_ingests_and_check_passes(store, temp_corpus):
    result = materialize(
        [corpus_source, pgvector_chunks, chunks_present],
        resources={
            "vector_store": VectorStoreResource(),
            "embedder": FakeEmbedderResource(),
        },
    )
    assert result.success

    evaluations = result.get_asset_check_evaluations()
    assert evaluations, "expected the chunks_present check to run"
    assert all(e.passed for e in evaluations)
