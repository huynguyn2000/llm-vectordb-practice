#!/usr/bin/env python3
"""
Interactive RAG REPL — ask questions against the sample document corpus.
Indexes documents once at startup; loops on stdin.

Usage:
  python repl.py

Commands:
  <question>        ask the RAG chatbot
  ?<question>       also print the retrieved sources with similarity scores
  q | quit | exit   leave
"""

from core.db import VectorStore
from core.embedder import Embedder
from use_cases.rag_chatbot import index_documents, retrieve, generate_answer
from data.documents import SAMPLE_DOCUMENTS


def main() -> None:
    with VectorStore() as store:
        embedder = Embedder()
        index_documents(SAMPLE_DOCUMENTS, store, embedder)

        print("\n=== RAG REPL ===")
        print("Ask a question. Prefix with '?' to also see retrieved sources. 'q' to quit.\n")

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in {"q", "quit", "exit"}:
                break

            show_sources = line.startswith("?")
            query = line[1:].strip() if show_sources else line
            if not query:
                continue

            docs = retrieve(query, store, embedder)
            if show_sources:
                print("\nRetrieved:")
                for d in docs:
                    snippet = d.content[:80] + ("…" if len(d.content) > 80 else "")
                    print(f"  [{d.score:.3f}] ({d.source}) {snippet}")
            answer = generate_answer(query, docs)
            print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
