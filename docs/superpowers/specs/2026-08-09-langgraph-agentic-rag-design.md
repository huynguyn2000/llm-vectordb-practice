# LangGraph Agentic RAG — Design

**Date:** 2026-08-09
**Status:** Approved (autonomous under session goal; open to course-correction)
**Context:** Capstone of the frameworks track. Builds a **LangGraph** agentic-RAG
graph (Corrective-RAG / CRAG-lite) on top of the LangChain components from the
`langchain_rag` package: retrieve → grade documents → (relevant? generate :
rewrite query and retry, capped). Dependencies are **injected** (retriever +
chat model), so the whole control flow — including the retry loop — is
deterministically testable with LangChain's fake chat models, no Ollama needed.

## Goals

- A `StateGraph` implementing CRAG-lite: retrieve, LLM-grade relevance, and if
  no relevant docs, rewrite the query and retry (bounded); otherwise generate.
- Reuse `langchain_rag.HybridRetriever` for retrieval and `ChatOllama` for the
  LLM in production.
- Inject retriever + LLM into `build_graph(...)` so tests drive every path
  (happy, retry-loop, exhausted) with `FakeListChatModel` + `FakeEmbedder`.
- A `demo.py agent "<query>"` subcommand.
- Local only, $0; LangSmith tracing still works (LangGraph runs are traced when
  the env vars are set).

## Non-goals

- Web-search fallback, multi-tool agents, human-in-the-loop, persistence /
  checkpointer (a bounded in-memory loop is enough for the capstone).
- Replacing the LCEL chain or the raw pipeline — this is an additional variant.
- Any cloud LLM.

## Architecture & layout

```
langgraph_rag/
  __init__.py
  state.py     # AgentState TypedDict
  nodes.py     # make_nodes(retriever, llm, ...) -> node callables (closures over deps)
  graph.py     # build_graph(retriever, llm, max_attempts=2) -> CompiledGraph; build_agent(store, embedder)
  tests/
    __init__.py
    test_graph.py         # compile/structure test (no infra)
    test_agent_flow.py    # end-to-end control flow with a fake retriever + FakeListChatModel (no infra)
demo.py        # + `agent "<query>"` subcommand
pyproject.toml # + langgraph in the langchain extra; langgraph_rag in packages
README.md      # + Agentic RAG (LangGraph) section
```

## State (`state.py`)

```python
class AgentState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    attempts: int
```

## Nodes (`nodes.py`) — closures over injected `retriever` and `llm`

`make_nodes(retriever, llm, max_attempts=2)` returns the node functions:

- **retrieve(state)** → `{"documents": retriever.invoke(state["question"]), "attempts": state.get("attempts", 0) + 1}`.
- **grade_documents(state)** → for each doc, ask `llm` "Is this document relevant
  to the question? Answer yes or no." Keep docs whose answer starts with `y`
  (case-insensitive). Return `{"documents": kept}`.
- **generate(state)** → format the (graded) docs with the shared citation format
  and the grounding prompt (answer only from context; else the refusal phrase),
  invoke `llm`, return `{"generation": text}`. With empty documents the prompt
  yields the refusal — the exhausted-retry path degrades gracefully.
- **transform_query(state)** → ask `llm` to rewrite `state["question"]` for
  better retrieval; return `{"question": rewritten}`.

## Graph (`graph.py`)

- Nodes wired: `START → retrieve → grade_documents → <decide> ...`.
- **Conditional edge** `decide_to_generate(state)`:
  - `documents` non-empty → `"generate"`.
  - else if `attempts >= max_attempts` → `"generate"` (graceful give-up → refusal).
  - else → `"transform_query"`.
- `transform_query → retrieve` (the corrective loop). `generate → END`.
- `build_graph(retriever, llm, max_attempts=2) -> CompiledGraph`.
- `build_agent(store, embedder, top_k=3, max_attempts=2)` wires a real
  `HybridRetriever` + `ChatOllama` (env-configured like the LCEL chain) and
  returns the compiled graph.

## Demo

`demo.py agent "<query>"`: builds the agent, invokes
`graph.invoke({"question": query, "documents": [], "generation": "", "attempts": 0})`,
prints `generation`. Requires Ollama.

## Error handling

- Postgres/Ollama down → node raises; error propagates (consistent with repo).
- Retry loop is bounded by `max_attempts` — cannot spin forever.
- Empty/irrelevant corpus → grading drops all docs → after `max_attempts` the
  generate node produces the refusal phrase.

## Testing

All graph tests are **no-infra** — a fake retriever (`RunnableLambda` returning a
fixed one-`Document` list) and `FakeListChatModel` inject deterministic
behavior, so the control flow is tested without Postgres or Ollama. (Real
retrieval is already covered by the `langchain_rag` retriever tests.) A fake
retriever returning exactly one doc means grading makes exactly one LLM call per
round, so the scripted response order is deterministic.

- `test_graph.py`: `build_graph(dummy_retriever, FakeListChatModel(responses=["yes"]))`
  compiles; the compiled graph exposes the expected nodes
  (`retrieve`, `grade_documents`, `generate`, `transform_query`).
- `test_agent_flow.py`:
  - **Happy path:** fake retriever returns one doc; scripted LLM responses
    `["yes", "<answer>"]` (grade relevant → generate); assert `generation ==
    "<answer>"` and `attempts == 1`.
  - **Retry loop:** scripted `["no", "<rewritten query>", "yes", "<answer>"]`
    (first grade rejects → transform → retrieve → grade accepts → generate);
    assert the graph looped (`attempts == 2`) and produced the answer. This
    exercises the conditional edge + corrective loop deterministically.
- Manual demo (`demo.py agent`) covers the real retriever + Ollama end-to-end.
- Existing suite untouched.

## Cost & resources

- **$0.** `langgraph` is free/OSS; models local. Tests use fakes (no Ollama).
- Reuses the existing Ollama + pgvector stack; LangGraph adds import overhead
  only.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Pattern | CRAG-lite (retrieve→grade→retry/generate) | Canonical agentic-RAG; shows conditional edges + loops (the point of LangGraph) |
| Dependency injection | `build_graph(retriever, llm)` | Deterministic testing of the whole flow with fakes |
| Grader | LLM yes/no per doc | Simple, demonstrates LLM-as-judge in a graph node |
| Loop bound | `max_attempts=2` | Prevents infinite corrective loops |
| Reuse | `HybridRetriever` + shared citation/grounding | Consistency with the LCEL variant; no new retrieval |
| Persistence | None (in-memory) | YAGNI for a local capstone |
