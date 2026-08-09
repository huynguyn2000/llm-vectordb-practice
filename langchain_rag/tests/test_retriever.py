import pytest
from langchain_core.documents import Document

from langchain_rag.retriever import HybridRetriever
from tests.helpers import FakeEmbedder, _purge_test_rows, fake_embedding

pytestmark = pytest.mark.integration

ROOT = "zz-test-lc"
PATH = "zz-test-lc/doc.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def test_retriever_returns_documents(store):
    store.upsert_source_with_chunks(
        ROOT, PATH, "test-hash", [("Photosynthesis converts sunlight.", 3, fake_embedding("x"))]
    )
    retriever = HybridRetriever(store=store, embedder=FakeEmbedder(), top_k=3)
    docs = retriever.invoke("photosynthesis")
    assert docs and all(isinstance(d, Document) for d in docs)
    ours = [d for d in docs if d.metadata.get("source_path") == PATH]
    assert ours, "expected the ingested chunk among results"
    assert set(ours[0].metadata) >= {"source_path", "chunk_index", "score", "id"}
