# LangChain RAG Pipeline + LangSmith Tracing — Design

**Date:** 2026-08-09
**Status:** Approved (design decisions made autonomously under the session goal; open to course-correction)
**Context:** Milestone in the "frameworks to learn" track. Rebuilds the existing
raw-Python RAG retrieval+generation on **LangChain** (as a comparison variant to
the hand-written pipeline) and wires **LangSmith** tracing. Everything stays
local (Ollama via `langchain-ollama`, pgvector) — no cloud API keys required to
run; LangSmith is env-gated and off by default. A follow-on milestone builds a
LangGraph agentic-RAG graph on top of these components.

## Goals

- A LangChain LCEL RAG chain that reuses the project's existing hybrid retrieval
  (vector + keyword + RRF) via a custom LangChain retriever — no re-ingestion,
  no new embeddings table.
- Local models through `langchain-ollama` (`ChatOllama`, `OllamaEmbeddings` not
  needed — retrieval reuses the existing `Embedder`).
- LangSmith tracing that activates purely from environment variables
  (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`), off by
  default, documented, with meaningful run names/tags on the chain.
- A `demo.py langchain "<query>"` subcommand.
- The grounding behavior mirrors the existing prompt (answer only from context;
  refuse otherwise).

## Non-goals

- Replacing the raw pipeline — this is a parallel variant for comparison.
- LangGraph / agentic flows (next milestone).
- LangChain's own PGVector store / re-ingestion — we wrap existing retrieval.
- Any cloud LLM. Ollama only.

## Why a custom retriever (not LangChain PGVector)

The project already has tested hybrid retrieval (`search.hybrid.hybrid_search`)
over the `chunks` table. Wrapping it in a `BaseRetriever` reuses that work,
avoids a second embeddings table, and still demonstrates real LangChain
integration. LangChain's `PGVector` would require re-ingesting into its own
schema — wasteful and off-message.

## Architecture & layout

```
langchain_rag/
  __init__.py
  retriever.py    # HybridRetriever(BaseRetriever) -> wraps VectorStore+Embedder, calls hybrid_search
  chain.py        # build_rag_chain(store, embedder, top_k=3) -> Runnable (LCEL); grounding prompt; LangSmith run-name/tags
  tests/
    __init__.py
    test_retriever.py   # integration: returns Documents from real Postgres (FakeEmbedder)
    test_chain.py       # build/structure test (no LLM call) + doc-formatting unit test
demo.py           # + `langchain "<query>"` subcommand
pyproject.toml    # + langchain extra (langchain, langchain-core, langchain-ollama, langsmith)
.env.example      # + LangSmith tracing vars (commented, off by default)
README.md         # + LangChain/LangSmith section
```

## Components

**`HybridRetriever`** (`retriever.py`) — a `langchain_core.retrievers.BaseRetriever`:
- Holds a `VectorStore`, an `Embedder`, and `top_k`.
- `_get_relevant_documents(query, *, run_manager)` calls
  `hybrid_search(query, store, embedder, top_k=top_k)` and maps each
  `ChunkResult` to a `Document(page_content=content, metadata={"source_path",
  "chunk_index", "score", "id"})`.
- Because `VectorStore`/`Embedder` aren't Pydantic-friendly, the retriever
  stores them as private attributes (constructed via a classmethod/factory
  `HybridRetriever.create(store, embedder, top_k=3)`), or declares
  `model_config = ConfigDict(arbitrary_types_allowed=True)`.

**RAG chain** (`chain.py`) — `build_rag_chain(store, embedder, top_k=3) -> Runnable`:
- LCEL: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | ChatOllama(...) | StrOutputParser()`.
- `format_docs` joins retrieved docs as `[Source: {source_path}#chunk{chunk_index}]\n{content}` — the same citation format as the raw pipeline.
- Prompt mirrors the existing grounding prompt (answer only from context; else
  "I don't know based on the provided documents.").
- `ChatOllama(model=os.getenv("LLM_MODEL","llama3.2"), base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0)`.
- The chain is tagged (`.with_config(run_name="langchain_rag", tags=["rag","hybrid"])`) so LangSmith traces are legible.

**LangSmith** — no code beyond env: LangChain auto-emits traces when
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set. `.env.example` gets
commented entries; README documents enabling. The chain's run-name/tags make the
traces meaningful. Off by default → the chain runs fully offline.

**Demo** — `demo.py langchain "<query>"`: builds the chain, invokes it, prints
the answer. Requires Ollama running.

## Error handling

- Empty query → `hybrid_search` returns `[]` → chain still runs; the prompt's
  refusal instruction yields the "I don't know" answer (matches raw behavior).
- Ollama/Postgres down → errors propagate (consistent with the rest of the repo).
- LangSmith key absent → tracing simply off; no error.

## Testing

- `test_retriever.py` (integration, real Postgres + `FakeEmbedder`): after
  ingesting a `zz-test-` doc, `HybridRetriever.create(...).invoke("<term>")`
  returns `Document`s whose metadata carries `source_path`/`chunk_index`; purge
  before/after. Marked `@pytest.mark.integration`.
- `test_chain.py` (no infra): `format_docs` produces the exact citation string
  for a list of `Document`s; `build_rag_chain(...)` returns a `Runnable` (assert
  it has `.invoke`) without calling the LLM. (End-to-end invoke is covered
  manually via the demo, since it needs Ollama.)
- Existing suite untouched.

## Cost & resources

- **$0.** LangChain/LangGraph/langsmith packages are free; models are local
  Ollama. LangSmith's cloud is optional and has a free tier, but tracing is off
  unless the user adds a key — no signup needed to run or test.
- RAM: unchanged (Ollama + pgvector). LangChain adds import overhead only.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Retrieval in LangChain | Custom `HybridRetriever` wrapping `hybrid_search` | Reuse tested hybrid search; no re-ingest |
| LLM | `langchain-ollama` `ChatOllama` | Matches local stack; no API key |
| Chain style | LCEL | Idiomatic modern LangChain |
| LangSmith | Env-gated tracing, off by default | No key needed to run; real integration when enabled |
| Prompt | Mirror existing grounding prompt | Fair comparison with the raw pipeline |
| Scope | Parallel variant, raw pipeline untouched | Build-then-compare story |
