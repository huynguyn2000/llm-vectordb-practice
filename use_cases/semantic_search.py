"""
Use case: Semantic Search over Documents
-----------------------------------------
Index a set of documents with their embeddings, then retrieve the most
semantically similar ones for a natural-language query.
"""

from core.db import VectorStore
from core.embedder import Embedder
from core.models import Document, DocumentResult


def index_documents(docs: list[Document], store: VectorStore, embedder: Embedder) -> None:
    store.clear_documents()
    for doc in docs:
        embedding = embedder.embed(doc.content)
        store.insert_document(doc.content, doc.source, embedding)
    print(f"Indexed {len(docs)} documents.")


def search(query: str, store: VectorStore, embedder: Embedder, top_k: int = 3) -> list[DocumentResult]:
    embedding = embedder.embed(query)
    rows = store.search_documents(embedding, top_k=top_k)
    return [DocumentResult(**r) for r in rows]


def run(store: VectorStore, embedder: Embedder) -> None:
    from data.documents import SAMPLE_DOCUMENTS

    print("\n=== Semantic Search ===")
    index_documents(SAMPLE_DOCUMENTS, store, embedder)

    queries = [
        "How does photosynthesis work?",
        "What is the capital of France?",
        "Explain machine learning in simple terms",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = search(query, store, embedder)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] ({r.source}) {r.content[:100]}...")
