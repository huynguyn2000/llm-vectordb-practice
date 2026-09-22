# Agent Patterns (LangGraph) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two canonical LangGraph agent patterns — query **routing** and **reflection** — as self-contained, dependency-injected graphs with deterministic fake-LLM tests.

**Architecture:** `agent_patterns/` package: `router.py` (classify → rag/direct/reject) and `reflection.py` (generate → reflect → revise, bounded). Each `build_*_graph(llm, ...)` takes an injected chat model, so control flow is tested with `FakeListChatModel` (no Ollama). `demo.py` gets `router`/`reflect` subcommands.

**Tech Stack:** Python 3.11+, langgraph, langchain-core, langchain-ollama, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-agent-patterns-design.md`

## Global Constraints

- `main` lacks `langgraph` — add `langgraph>=0.2,<0.6` to the existing `langchain` extra. No new base deps.
- Each graph built via `build_*_graph(llm, ...)`; graphs are self-contained (no dependency on `langgraph_rag`, core DB, or retrieval).
- Router: unrecognized classify label → default `"direct"` (never KeyError). Reflection: bounded by `max_revisions`.
- All tests no-infra via `FakeListChatModel` (scripted responses matching the node call order). No Ollama in tests.
- Add `agent_patterns`, `agent_patterns.tests` to `[tool.setuptools] packages`.
- Every commit ends with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Router + Reflection graphs + tests

**Files:**
- Modify: `pyproject.toml` (langgraph in langchain extra + packages)
- Create: `agent_patterns/__init__.py` (empty), `router.py`, `reflection.py`
- Create: `agent_patterns/tests/__init__.py` (empty), `test_router.py`, `test_reflection.py`

**Interfaces:**
- Produces: `build_router_graph(llm) -> CompiledGraph` + `RouterState` (agent_patterns.router); `build_reflection_graph(llm, max_revisions=2) -> CompiledGraph` + `ReflectionState` (agent_patterns.reflection). Task 2 consumes both.

- [ ] **Step 1: pyproject** — add `langgraph>=0.2,<0.6` to the `langchain` extra list; add `agent_patterns`, `agent_patterns.tests` to `[tool.setuptools] packages`. Then `pip install -e ".[dev,langchain]"` (langgraph already installed; confirm `.venv/bin/pytest --collect-only -q` has no errors).

- [ ] **Step 2: package markers** — `mkdir -p agent_patterns/tests && touch agent_patterns/__init__.py agent_patterns/tests/__init__.py`

- [ ] **Step 3: `agent_patterns/router.py`**

```python
"""LangGraph query-routing agent: classify the question, then route to a
rag / direct / reject handler."""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

_ROUTES = {"rag", "direct", "reject"}

_CLASSIFY = ChatPromptTemplate.from_template(
    "Classify the user question into exactly one label: rag, direct, or reject.\n"
    "- rag: needs looking up documents or knowledge\n"
    "- direct: a general question answerable directly\n"
    "- reject: harmful, nonsensical, or out of scope\n"
    "Answer with ONLY the label.\n\nQuestion: {question}"
)
_ANSWER = ChatPromptTemplate.from_template(
    "Answer the question.\n\nQuestion: {question}\nAnswer:"
)


class RouterState(TypedDict):
    question: str
    route: str
    answer: str


def build_router_graph(llm):
    classify_chain = _CLASSIFY | llm | StrOutputParser()
    answer_chain = _ANSWER | llm | StrOutputParser()

    def classify(state: RouterState) -> dict:
        label = classify_chain.invoke({"question": state["question"]}).strip().lower()
        return {"route": label if label in _ROUTES else "direct"}

    def rag_node(state: RouterState) -> dict:
        # In practice this calls the RAG pipeline; here the LLM answers as the rag path.
        return {"answer": "[rag] " + answer_chain.invoke({"question": state["question"]})}

    def direct_node(state: RouterState) -> dict:
        return {"answer": "[direct] " + answer_chain.invoke({"question": state["question"]})}

    def reject_node(state: RouterState) -> dict:
        return {"answer": "I can't help with that request."}

    g = StateGraph(RouterState)
    g.add_node("classify", classify)
    g.add_node("rag", rag_node)
    g.add_node("direct", direct_node)
    g.add_node("reject", reject_node)
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", lambda s: s["route"],
        {"rag": "rag", "direct": "direct", "reject": "reject"},
    )
    g.add_edge("rag", END)
    g.add_edge("direct", END)
    g.add_edge("reject", END)
    return g.compile()
```

- [ ] **Step 4: `agent_patterns/reflection.py`**

```python
"""LangGraph reflection agent: generate a draft, critique it, and revise until
approved or a bounded number of revisions."""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

_GENERATE = ChatPromptTemplate.from_template(
    "Write a short response to the task.\n\nTask: {task}\nResponse:"
)
_REFLECT = ChatPromptTemplate.from_template(
    "Critique the draft for the task. If it is good enough, reply with exactly "
    "'APPROVE'. Otherwise give one concrete improvement.\n\n"
    "Task: {task}\n\nDraft: {draft}\n\nCritique:"
)
_REVISE = ChatPromptTemplate.from_template(
    "Revise the draft using the critique.\n\nTask: {task}\n\nDraft: {draft}\n\n"
    "Critique: {critique}\n\nRevised:"
)


class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    revisions: int
    approved: bool


def build_reflection_graph(llm, max_revisions: int = 2):
    gen = _GENERATE | llm | StrOutputParser()
    reflect_chain = _REFLECT | llm | StrOutputParser()
    revise_chain = _REVISE | llm | StrOutputParser()

    def generate(state: ReflectionState) -> dict:
        return {"draft": gen.invoke({"task": state["task"]}), "revisions": 0, "approved": False}

    def reflect(state: ReflectionState) -> dict:
        c = reflect_chain.invoke({"task": state["task"], "draft": state["draft"]}).strip()
        return {"critique": c, "approved": c.upper().startswith("APPROVE")}

    def revise(state: ReflectionState) -> dict:
        new = revise_chain.invoke(
            {"task": state["task"], "draft": state["draft"], "critique": state["critique"]}
        )
        return {"draft": new, "revisions": state["revisions"] + 1}

    def decide(state: ReflectionState) -> str:
        if state["approved"] or state["revisions"] >= max_revisions:
            return "end"
        return "revise"

    g = StateGraph(ReflectionState)
    g.add_node("generate", generate)
    g.add_node("reflect", reflect)
    g.add_node("revise", revise)
    g.add_edge(START, "generate")
    g.add_edge("generate", "reflect")
    g.add_conditional_edges("reflect", decide, {"revise": "revise", "end": END})
    g.add_edge("revise", "reflect")
    return g.compile()
```

- [ ] **Step 5: `agent_patterns/tests/test_router.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent_patterns.router import build_router_graph


def _run(llm, q):
    return build_router_graph(llm).invoke({"question": q, "route": "", "answer": ""})


def test_routes_to_rag():
    r = _run(FakeListChatModel(responses=["rag", "an answer"]), "look this up in the docs")
    assert r["route"] == "rag"
    assert r["answer"].startswith("[rag]")


def test_routes_to_direct():
    r = _run(FakeListChatModel(responses=["direct", "2 plus 2 is 4"]), "what is 2+2")
    assert r["route"] == "direct"
    assert r["answer"].startswith("[direct]")


def test_routes_to_reject():
    r = _run(FakeListChatModel(responses=["reject"]), "do something harmful")
    assert r["route"] == "reject"
    assert "can't help" in r["answer"].lower()


def test_unknown_label_defaults_to_direct():
    r = _run(FakeListChatModel(responses=["banana", "fallback"]), "ambiguous")
    assert r["route"] == "direct"
    assert r["answer"].startswith("[direct]")
```

- [ ] **Step 6: `agent_patterns/tests/test_reflection.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent_patterns.reflection import build_reflection_graph


def _run(llm, task, max_revisions=2):
    return build_reflection_graph(llm, max_revisions=max_revisions).invoke(
        {"task": task, "draft": "", "critique": "", "revisions": 0, "approved": False}
    )


def test_approved_first_pass():
    # generate -> "draft one"; reflect -> "APPROVE"
    r = _run(FakeListChatModel(responses=["draft one", "APPROVE"]), "task")
    assert r["approved"] is True
    assert r["revisions"] == 0
    assert r["draft"] == "draft one"


def test_one_revision_then_approve():
    # generate -> d1; reflect -> critique; revise -> d2; reflect -> APPROVE
    r = _run(FakeListChatModel(responses=["draft one", "improve X", "draft two", "APPROVE"]), "task")
    assert r["revisions"] == 1
    assert r["draft"] == "draft two"
    assert r["approved"] is True


def test_stops_at_max_revisions():
    # never approves: generate, reflect, revise, reflect, revise, reflect -> cap
    r = _run(FakeListChatModel(responses=["d0", "crit", "d1", "crit", "d2", "crit"]), "task", max_revisions=2)
    assert r["revisions"] == 2
    assert r["approved"] is False
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/pytest agent_patterns/tests/ -v`
Expected: 7 pass (no infra). Then `.venv/bin/pytest -q` full suite — no regressions/collection errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml agent_patterns/
git commit -m "feat: LangGraph agent patterns — router + reflection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Demo subcommands + docs

**Files:**
- Modify: `demo.py` (add `router` + `reflect` subcommands)
- Modify: `README.md`

**Interfaces:**
- Consumes: `build_router_graph`, `build_reflection_graph` (Task 1); `ChatOllama`.

- [ ] **Step 1: Add subcommands to `demo.py`** (after the last existing subcommand branch, before the USE_CASES validation):

```python
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
```

- [ ] **Step 2: Manual verify (needs Ollama)**

```bash
docker compose up -d
.venv/bin/python demo.py router "what is machine learning"
.venv/bin/python demo.py reflect "write a one-sentence tagline for a vector database"
```
Expected: router prints a route + answer; reflect prints revisions/approved/draft. Capture both. If Ollama is slow, `timeout` them; the fake-LLM unit tests are the gate.

- [ ] **Step 3: Update `README.md`**

1. Use-case rows: `router "<q>"` (route + answer) and `reflect "<task>"` (generate→critique→revise).
2. Stack bullet: `- **Agent patterns**: LangGraph graphs — query routing and reflection (generate→critique→revise) — alongside the CRAG-lite corrective RAG; ReAct/supervisor are documented next patterns`.
3. Project Structure: `agent_patterns/` block (router.py, reflection.py).
4. A short **Agent patterns** section: the two graphs, `demo.py router/reflect`, and a note that ReAct (tool-calling) and supervisor/multi-agent are the natural next patterns (need a tool-calling model, so not built here).

- [ ] **Step 4: Final verification**

Run: `.venv/bin/pytest -q`
Expected: full suite passes, no collection errors.

- [ ] **Step 5: Commit**

```bash
git add demo.py README.md
git commit -m "feat: router/reflect demos + document agent patterns

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
