"""A LangChain retriever that reuses the project's hybrid search.

Wrapping the existing hybrid_search (vector + keyword + RRF) keeps a single
source of retrieval truth and avoids a second embeddings table."""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, SkipValidation

from core.db import VectorStore
from core.embedder import Embedder
from search.hybrid import hybrid_search


class HybridRetriever(BaseRetriever):
    """Retrieve chunks via hybrid_search, as LangChain Documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: SkipValidation[VectorStore]
    embedder: SkipValidation[Embedder]
    top_k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = hybrid_search(query, self.store, self.embedder, top_k=self.top_k)
        return [
            Document(
                page_content=r.content,
                metadata={
                    "source_path": r.source_path,
                    "chunk_index": r.chunk_index,
                    "score": r.score,
                    "id": r.id,
                },
            )
            for r in results
        ]
