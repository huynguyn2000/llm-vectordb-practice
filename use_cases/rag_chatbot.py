"""
Use case: RAG Chatbot
----------------------
Retrieve relevant document chunks from the vector store and pass them
as context to a local Ollama LLM to generate a grounded answer.
"""

import os
import ollama
from dotenv import load_dotenv

from core.db import VectorStore
from core.embedder import Embedder
from core.models import Document, DocumentResult

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")


def index_documents(docs: list[Document], store: VectorStore, embedder: Embedder) -> None:
    store.clear_documents()
    for doc in docs:
        embedding = embedder.embed(doc.content)
        store.insert_document(doc.content, doc.source, embedding)
    print(f"Indexed {len(docs)} documents.")


def retrieve(query: str, store: VectorStore, embedder: Embedder, top_k: int = 3) -> list[DocumentResult]:
    embedding = embedder.embed(query)
    rows = store.search_documents(embedding, top_k=top_k)
    return [DocumentResult(**r) for r in rows]


def generate_answer(query: str, context_docs: list[DocumentResult]) -> str:
    context = "\n\n".join(
        f"[Source: {d.source}]\n{d.content}" for d in context_docs
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know based on the provided documents."

        Context:
        {context}

        Question: {query}
        Answer:
    """

    client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    response = client.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"].strip()


def chat(query: str, store: VectorStore, embedder: Embedder) -> str:
    docs = retrieve(query, store, embedder)
    return generate_answer(query, docs)


def run(store: VectorStore, embedder: Embedder) -> None:
    from data.documents import SAMPLE_DOCUMENTS

    print("\n=== RAG Chatbot ===")
    index_documents(SAMPLE_DOCUMENTS, store, embedder)

    questions = [
        "What is machine learning?",
        "How do plants make food?",
        "What is the speed of light?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = chat(q, store, embedder)
        print(f"Answer:   {answer}")
