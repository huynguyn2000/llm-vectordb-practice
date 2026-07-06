"""
Use case: RAG Chatbot
----------------------
Ingest the local corpus (chunked + embedded), retrieve the most relevant
chunks from pgvector, and pass them as context to a local Ollama LLM to
generate a grounded answer.
"""

import os

import ollama
from dotenv import load_dotenv

from core.db import VectorStore
from core.embedder import Embedder
from core.models import ChunkResult
from ingestion.ingest import ingest_directory

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
CORPUS_DIR = "data/corpus"


def retrieve(
    query: str, store: VectorStore, embedder: Embedder, top_k: int = 3
) -> list[ChunkResult]:
    embedding = embedder.embed(query)
    rows = store.search_chunks(embedding, top_k=top_k)
    return [ChunkResult(**r) for r in rows]


def generate_answer(query: str, context_chunks: list[ChunkResult]) -> str:
    context = "\n\n".join(
        f"[Source: {c.source_path}#chunk{c.chunk_index}]\n{c.content}"
        for c in context_chunks
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
    chunks = retrieve(query, store, embedder)
    return generate_answer(query, chunks)


def run(store: VectorStore, embedder: Embedder) -> None:
    print("\n=== RAG Chatbot ===")
    stats = ingest_directory(CORPUS_DIR, store, embedder)
    print(
        f"Corpus ready: {stats.ingested} ingested, {stats.skipped} unchanged, "
        f"{stats.deleted} deleted, {stats.failed} failed."
    )

    questions = [
        "What is machine learning?",
        "How do plants make food?",
        "What is the speed of light?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = chat(q, store, embedder)
        print(f"Answer:   {answer}")
