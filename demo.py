#!/usr/bin/env python3
"""
Vector DB Practice — Demo Runner
Usage:
  python demo.py                    # run all use cases
  python demo.py semantic           # semantic search only
  python demo.py rag                # RAG chatbot only
  python demo.py products           # product similarity only
  python demo.py logs               # log anomaly clustering only
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
