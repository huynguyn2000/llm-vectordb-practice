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

    if selected == "vsdb":
        if len(sys.argv) < 3:
            print('Usage: python demo.py vsdb "<query>"')
            return
        try:
            from vectorstores.corpus import build_chroma_backend_from_corpus
            from vectorstores.pgvector import PgvectorBackend
        except ImportError:
            print("Install the chroma extra: pip install -e '.[chroma]'")
            return
        query = sys.argv[2]
        with VectorStore() as store:
            embedder = Embedder()
            emb = embedder.embed(query)
            pg = PgvectorBackend(store).search(emb, top_k=5)
            chroma = build_chroma_backend_from_corpus("data/corpus", embedder).search(emb, top_k=5)

        def label(rows, i):
            if i >= len(rows):
                return ""
            r = rows[i]
            return f"{r['source_path']}#chunk{r['chunk_index']} ({r['score']:.3f})"

        print(f'\nQuery: "{query}"\n')
        print(f" {'rank':<5}{'pgvector':<40}{'chroma'}")
        for i in range(max(len(pg), len(chroma))):
            print(f" {i + 1:<5}{label(pg, i):<40}{label(chroma, i)}")
        return

    if selected == "langchain":
        from langchain_rag.chain import build_rag_chain

        if len(sys.argv) < 3:
            print('Usage: python demo.py langchain "<query>"')
            return
        query = sys.argv[2]
        with VectorStore() as store:
            chain = build_rag_chain(store, Embedder())
            answer = chain.invoke(query)
        print(f"\nQuestion: {query}\nAnswer:   {answer}")
        return

    if selected in ("router", "reflect"):
        if len(sys.argv) < 3:
            print(f'Usage: python demo.py {selected} "<text>"')
            return
        import os

        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
        text = sys.argv[2]
        if selected == "router":
            from agent_patterns.router import build_router_graph

            r = build_router_graph(llm).invoke({"question": text, "route": "", "answer": ""})
            print(f"route: {r['route']}\nanswer: {r['answer']}")
        else:
            from agent_patterns.reflection import build_reflection_graph

            r = build_reflection_graph(llm).invoke(
                {"task": text, "draft": "", "critique": "", "revisions": 0, "approved": False}
            )
            print(f"revisions: {r['revisions']}\napproved: {r['approved']}\ndraft: {r['draft']}")
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
