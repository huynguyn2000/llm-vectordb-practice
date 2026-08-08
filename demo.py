#!/usr/bin/env python3
"""
Vector DB Practice — Demo Runner
Usage:
  python demo.py                    # run all use cases
  python demo.py semantic           # semantic search only
  python demo.py rag                # RAG chatbot only
  python demo.py products           # product similarity only
  python demo.py logs               # log anomaly clustering only
  python demo.py ingest [dir]       # ingest a folder of .md/.txt/.pdf (default: data/corpus)
"""

import sys
from core.db import VectorStore
from core.embedder import Embedder

USE_CASES = {
    "semantic": ("Semantic Search", "use_cases.semantic_search"),
    "rag": ("RAG Chatbot", "use_cases.rag_chatbot"),
    "products": ("Product Similarity", "use_cases.product_similarity"),
    "logs": ("Log / Anomaly Clustering", "use_cases.log_clustering"),
}


def main():
    selected = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if selected == "ingest":
        from ingestion.ingest import ingest_directory

        corpus_dir = sys.argv[2] if len(sys.argv) > 2 else "data/corpus"
        with VectorStore() as store:
            stats = ingest_directory(corpus_dir, store, Embedder())
        print(
            f"Ingested: {stats.ingested}  skipped (unchanged): {stats.skipped}  "
            f"deleted: {stats.deleted}  failed: {stats.failed}"
        )
        return

    if selected == "compare":
        from use_cases.rag_chatbot import retrieve, retrieve_vector

        if len(sys.argv) < 3:
            print('Usage: python demo.py compare "<query>"')
            return
        query = sys.argv[2]
        with VectorStore() as store:
            embedder = Embedder()
            vec = retrieve_vector(query, store, embedder, top_k=5)
            hyb = retrieve(query, store, embedder, top_k=5)

        def label(chunks, i):
            if i >= len(chunks):
                return ""
            c = chunks[i]
            return f"{c.source_path}#chunk{c.chunk_index}"

        print(f'\nQuery: "{query}"\n')
        print(f" {'rank':<5}{'vector-only':<34}{'hybrid (RRF)'}")
        for i in range(max(len(vec), len(hyb))):
            print(f" {i + 1:<5}{label(vec, i):<34}{label(hyb, i)}")
        return

    if selected not in USE_CASES and selected != "all":
        print(f"Unknown use case: {selected}")
        print(f"Available: {', '.join(USE_CASES)} or 'all'")
        sys.exit(1)

    keys = list(USE_CASES.keys()) if selected == "all" else [selected]

    with VectorStore() as store:
        embedder = Embedder()
        for key in keys:
            import importlib
            mod = importlib.import_module(USE_CASES[key][1])
            mod.run(store, embedder)
            print()


if __name__ == "__main__":
    main()
