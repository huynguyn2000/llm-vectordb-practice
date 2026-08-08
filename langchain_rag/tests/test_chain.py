import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from langchain_rag.chain import build_rag_chain, format_docs
from tests.helpers import FakeEmbedder, _purge_test_rows, fake_embedding


def test_format_docs_citation():
    docs = [
        Document(page_content="Alpha.", metadata={"source_path": "a.md", "chunk_index": 0}),
        Document(page_content="Beta.", metadata={"source_path": "b.md", "chunk_index": 2}),
    ]
    assert format_docs(docs) == (
        "[Source: a.md#chunk0]\nAlpha.\n\n[Source: b.md#chunk2]\nBeta."
    )


@pytest.mark.integration
def test_build_rag_chain_is_runnable(store):
    # Constructing the chain opens a DB connection (VectorStore) but does NOT
    # call the LLM, so no Ollama is needed here.
    _purge_test_rows(store)
    chain = build_rag_chain(store, FakeEmbedder(), top_k=3)
    assert isinstance(chain, Runnable)
    assert hasattr(chain, "invoke")
