# Mem0 Persistent Memory — Design

**Date:** 2026-08-10
**Status:** Approved (autonomous execution under session goal)
**Context:** Adds a persistent **memory layer** (Mem0) so a chat/agent can
remember facts about a user across turns/sessions — e.g. preferences stated
earlier resurface in later answers. Delivered as a thin, local-first wrapper:
Mem0 configured with **local Ollama** (LLM + embeddings) and a local vector
store, exposed via a small `remember()` / `recall()` API and a demo. Opt-in
`memory` extra; lazy import; no-infra config test.

## Honest risk note (baked into scope)

Mem0 (a) pulls a sizeable dependency tree (conflict risk like Docling), and
(b) its memory *extraction* is LLM-driven — on local `llama3.2` that is slow
(like the Ragas judge). So the design is defensive: the wrapper + config are
the deliverable and are unit-testable without Mem0 installed; a real end-to-end
memory demo is best-effort and **documented-if-heavy** (install conflict or slow
extraction → note it, recommend a separate venv / API LLM), exactly as the
Ragas/Docling milestones handled their environmental limits.

## Goals

- `memory/` package: `build_memory()` → a Mem0 `Memory` configured for local
  Ollama (LLM + embedder) + a local vector store; `remember(mem, user_id,
  messages)` and `recall(mem, user_id, query)` helpers returning plain data.
- `demo.py memory "<user_id>"` — a tiny scripted scenario: store a couple of
  facts, then recall them for a query (shows cross-turn memory).
- Opt-in `memory` extra (`mem0ai`); Mem0 imported lazily so the package loads
  and the config helper is unit-testable without it.
- No changes to core/search/ingestion/use_cases.

## Non-goals

- Wiring Mem0 into the LangGraph agent automatically (documented as the natural
  next step; kept separate to keep this milestone small and testable).
- Cloud Mem0 platform / API keys.
- A deterministic end-to-end memory unit test (extraction is LLM-driven; covered
  by the manual demo + a config-shape unit test).

## Architecture & layout

```
memory/
  __init__.py
  config.py      # build_memory_config() -> dict (local Ollama LLM + embedder + vector store); pure/unit-testable
  store.py       # build_memory() -> Memory (lazy mem0 import); remember()/recall() helpers
  tests/
    __init__.py
    test_config.py     # no-infra: config dict shape (provider=ollama, models, vector store) — no mem0 import
demo.py          # + `memory "<user_id>"` subcommand
pyproject.toml   # + `memory` extra: mem0ai
README.md        # + Memory (Mem0) section incl. the heavy/slow + separate-venv caveat
```

## Config (`config.py`) — the unit-testable core

`build_memory_config() -> dict` returns Mem0's config dict for a fully-local
setup, from env with sensible defaults:

```python
{
  "llm": {"provider": "ollama",
          "config": {"model": LLM_MODEL, "ollama_base_url": OLLAMA_BASE_URL}},
  "embedder": {"provider": "ollama",
               "config": {"model": EMBED_MODEL, "ollama_base_url": OLLAMA_BASE_URL}},
  "vector_store": {"provider": "chroma",
                   "config": {"collection_name": "mem0", "path": ".mem0_chroma"}},
}
```

This is a pure function (no Mem0 import) → the unit test asserts its shape
(providers are `ollama`/`ollama`/`chroma`, model names from env) with zero infra.
(Exact key names verified against the installed mem0ai during implementation;
adapt to the installed version if they differ, and note it.)

## Store (`store.py`)

- `build_memory()` — lazily `from mem0 import Memory`;
  `Memory.from_config(build_memory_config())`. Raises a clear error naming the
  `memory` extra if mem0 isn't installed.
- `remember(mem, user_id, messages)` — `mem.add(messages, user_id=user_id)`.
- `recall(mem, user_id, query)` — `mem.search(query, user_id=user_id)` → return
  the list of memory strings.

(Mem0's exact method signatures/return shapes are version-sensitive; the
implementer verifies against the installed mem0ai and adapts, documenting any
difference — same posture as the Ragas EvaluationResult extraction.)

## Demo (`demo.py memory "<user_id>"`)

Scripted: `remember` two facts (e.g. "I prefer concise answers", "I work with
pgvector"), then `recall` for a query ("what do you know about me?") and print
the retrieved memories. Requires Ollama + the `memory` extra.

## Testing

- `test_config.py` (**no infra, no mem0**): `build_memory_config()` returns the
  expected provider/model/vector-store shape. Pure — the reliable gate.
- Manual/integration (best-effort): `pip install -e ".[memory]"`, run
  `demo.py memory demo-user`, confirm recalled memories reflect what was stored.
  **If mem0ai install conflicts with the stack or local extraction is too slow,
  document it** (code + config are the deliverable; recommend a separate venv or
  an API LLM for real use) — do not block the milestone.
- Existing suite untouched.

## Cost & resources

- **$0** local. `memory` extra pulls `mem0ai` (+ its deps) — potentially heavy;
  kept opt-in so the default env is unaffected. Local memory extraction is slow
  on `llama3.2`; an API LLM is faster for real use (documented).

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Memory lib | Mem0 (`mem0ai`), local config | The user's named stack; local-first keeps it $0 |
| Delivery | Opt-in `memory` extra, lazy import | Default env lean; dep-conflict containment (Docling lesson) |
| Testable core | `build_memory_config()` pure function | Deterministic no-infra gate; extraction is LLM-driven and not unit-tested |
| Vector store | Chroma (embedded) for Mem0 | Reuses the local, no-Docker store from the VDB milestone |
| Agent wiring | Documented next step, not built | Keeps this milestone small + testable |
| Full e2e run | Best-effort, documented-if-heavy | Mem0 install/extraction friction expected (Ragas/Docling posture) |
