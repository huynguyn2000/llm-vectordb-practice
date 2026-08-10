# Agent Patterns (LangGraph) — Design

**Date:** 2026-08-10
**Status:** Approved (autonomous execution under session goal)
**Context:** Beyond the CRAG-lite corrective-RAG capstone, this adds two more
canonical LangGraph agent patterns as self-contained graphs: **query routing**
and **reflection (generate → critique → revise)**. Both use dependency-injected
chat models so the whole control flow is deterministically testable with
`FakeListChatModel` — no Ollama needed for tests. Demonstrates breadth of agent
design, not just one pattern.

## Goals

- `agent_patterns/` package with two graphs, each `build_*_graph(llm, ...) -> CompiledGraph`:
  - **Router**: classify the query → route to `rag` / `direct` / `reject`.
  - **Reflection**: generate a draft → critique → revise until approved or a
    bounded number of revisions.
- Deterministic control-flow tests (fake LLM), like the CRAG capstone.
- `demo.py router "<q>"` and `demo.py reflect "<task>"` (real Ollama).
- Self-contained: does not depend on `langgraph_rag`; only `langgraph` +
  `langchain-core` + `langchain-ollama`.

## Non-goals

- ReAct tool-calling agent and supervisor/multi-agent: documented as further
  patterns, not built here — deterministic testing needs a tool-calling-capable
  model (FakeListChatModel can't emit tool calls), which would make the tests
  flaky/slow. Noted in the README as next patterns.
- Wiring routing's `rag` branch to real retrieval (kept LLM/text-level so the
  pattern is the focus and tests stay infra-free).

## Dependencies

- `main` currently lacks `langgraph`. Add `langgraph>=0.2,<0.6` to the existing
  `langchain` extra (same pin used by the langgraph_rag capstone). No new base
  deps.

## Architecture & layout

```
agent_patterns/
  __init__.py
  router.py       # build_router_graph(llm) -> CompiledGraph; RouterState
  reflection.py   # build_reflection_graph(llm, max_revisions=2) -> CompiledGraph; ReflectionState
  tests/
    __init__.py
    test_router.py       # no-infra: FakeListChatModel drives each route
    test_reflection.py   # no-infra: FakeListChatModel drives the critique/revise loop
demo.py           # + `router "<q>"` and `reflect "<task>"` subcommands
pyproject.toml    # + langgraph in the langchain extra
README.md         # + Agent patterns section (+ ReAct/supervisor as next)
```

## Router (`router.py`)

`RouterState = {question: str, route: str, answer: str}`.
- `classify(state)`: `llm` classifies the question into exactly one of
  `rag` / `direct` / `reject` (prompt returns the bare label); parse to `route`
  (default `direct` if the label isn't recognized).
- Conditional edge on `route` → `rag_node` / `direct_node` / `reject_node`.
- Terminal nodes set `answer`: `rag_node` (would call retrieval in practice;
  here uses `llm` to answer, labeled as the RAG path), `direct_node` (`llm`
  answers directly), `reject_node` (fixed refusal string). Each → END.

## Reflection (`reflection.py`)

`ReflectionState = {task: str, draft: str, critique: str, revisions: int, approved: bool}`.
- `generate(state)`: `llm` produces the first `draft`.
- `reflect(state)`: `llm` critiques the draft; if the critique starts with
  `APPROVE` (case-insensitive) → `approved=True`, else store the critique.
- `decide`: `approved` or `revisions >= max_revisions` → END; else → `revise`.
- `revise(state)`: `llm` rewrites the draft using the critique; `revisions += 1`
  → back to `reflect`.

Bounded loop (`max_revisions`) — cannot spin forever.

## Demo

- `demo.py router "<question>"` → prints the chosen route + answer.
- `demo.py reflect "<task>"` → prints the final draft + revision count.
Both build the graph with a real `ChatOllama` (env-configured) and invoke it.

## Testing

All no-infra, `FakeListChatModel` scripts the LLM outputs deterministically:
- **Router**: three tests — classify→`rag`/`direct`/`reject` each route to the
  right terminal node and set `answer`; plus an unrecognized label → defaults to
  `direct`. Assert `route` and that the expected node ran.
- **Reflection**: (a) approve on first reflect → `revisions == 0`, final draft is
  the first draft; (b) critique then approve → loops once (`revisions == 1`),
  final draft is the revised one; (c) never approves → stops at `max_revisions`.
  Response ordering scripted to match the node call sequence.
- Existing suite untouched.

## Error handling

- Unrecognized route label → safe default (`direct`), never a KeyError.
- Reflection loop bounded by `max_revisions`.
- Ollama/infra only touched by the demos, not the tests.

## Cost & resources

- **$0.** Tests are fake-LLM (no Ollama). Demos use local Ollama. `langgraph`
  is already installed; only the pyproject declaration is added.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Patterns | Router + Reflection | Canonical, and text-level → deterministically testable with FakeListChatModel |
| ReAct / supervisor | Documented, not built | Need tool-calling model; FakeListChatModel can't emit tool calls → flaky tests |
| Dependency injection | `build_*_graph(llm, ...)` | Same testable pattern as the CRAG capstone |
| Router rag-branch | LLM/text-level (not real retrieval) | Keeps the pattern the focus + tests infra-free |
| langgraph dep | Add to the langchain extra | main lacks it (stacked-merge gap); same pin as capstone |
