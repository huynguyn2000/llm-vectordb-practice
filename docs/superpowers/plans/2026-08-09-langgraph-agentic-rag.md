# LangGraph Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangGraph CRAG-lite agent (retrieve → grade → retry/generate) reusing the LangChain components, with dependency injection so the whole control flow is deterministically testable without infra.

**Architecture:** New `langgraph_rag/` package: `state.py` (TypedDict), `nodes.py` (`make_nodes(retriever, llm)` closures + `make_decider`), `graph.py` (`build_graph(retriever, llm)` compiles a `StateGraph`; `build_agent(store, embedder)` wires the real `HybridRetriever` + `ChatOllama`). A `demo.py agent` subcommand runs it.

**Tech Stack:** Python 3.11+, LangGraph, LangChain (`langchain-core`, `langchain-ollama`, existing `langchain_rag`), pytest with `FakeListChatModel`.

**Spec:** `docs/superpowers/specs/2026-08-09-langgraph-agentic-rag-design.md`

## Global Constraints

- Branch is `feat/langgraph-agent`, stacked on `feat/langchain-rag` (it imports `langchain_rag.retriever.HybridRetriever`).
- Dependencies injected: `build_graph(retriever, llm, max_attempts=2)`; `build_agent` wires real `HybridRetriever(store, embedder, top_k)` + `ChatOllama(model=os.getenv("LLM_MODEL","llama3.2"), base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"), temperature=0)`.
- `AgentState` = TypedDict with `question: str`, `documents: list[Document]`, `generation: str`, `attempts: int`.
- Conditional edge `decide_to_generate`: docs non-empty → `"generate"`; else attempts ≥ max → `"generate"`; else `"transform_query"`. `transform_query → retrieve`; `generate → END`.
- Grounding prompt/citation format consistent with the rest of the repo (answer only from context; else exactly `I don't know based on the provided documents.`).
- All tests are no-infra (fake retriever + `FakeListChatModel`); no Postgres/Ollama in the suite. No `@pytest.mark.integration` needed here.
- No changes to `core/`, `ingestion/`, `search/`, `langchain_rag/`, or the raw pipeline.
- Run from repo root with the venv: `.venv/bin/pytest`. Install with `pip install -e ".[dev,langchain]"` (langgraph added to that extra).
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Deps + state + nodes + graph + compile test

**Files:**
- Modify: `pyproject.toml` (add `langgraph` to the `langchain` extra; add `langgraph_rag` to packages)
- Create: `langgraph_rag/__init__.py` (empty), `langgraph_rag/state.py`, `langgraph_rag/nodes.py`, `langgraph_rag/graph.py`
- Create: `langgraph_rag/tests/__init__.py` (empty)
- Test: `langgraph_rag/tests/test_graph.py`

**Interfaces:**
- Produces: `AgentState` (`langgraph_rag.state`); `make_nodes(retriever, llm, max_attempts=2) -> dict`, `make_decider(max_attempts=2) -> callable` (`langgraph_rag.nodes`); `build_graph(retriever, llm, max_attempts=2) -> CompiledStateGraph`, `build_agent(store, embedder, top_k=3, max_attempts=2)` (`langgraph_rag.graph`). Task 2 consumes `build_graph`.

- [ ] **Step 1: Add langgraph dep + package to `pyproject.toml`**

Append `langgraph` to the existing `langchain` extra so it reads:

```toml
langchain = ["langchain>=0.3,<0.4", "langchain-ollama>=0.2", "langsmith>=0.1", "langgraph>=0.2,<0.6"]
```

Add `langgraph_rag` to the packages list (keep the existing entries, which on this branch are `core, use_cases, data, ingestion, search, langchain_rag`):

```toml
packages = ["core", "use_cases", "data", "ingestion", "search", "langchain_rag", "langgraph_rag"]
```

- [ ] **Step 2: Install**

```bash
pip install -e ".[dev,langchain]"
```

Expected: langgraph installs.

- [ ] **Step 3: Create package markers**

```bash
mkdir -p langgraph_rag/tests && touch langgraph_rag/__init__.py langgraph_rag/tests/__init__.py
```

- [ ] **Step 4: Create `state.py`**

```python
"""Agent state shared across LangGraph nodes."""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    attempts: int
```

- [ ] **Step 5: Create `nodes.py`**

```python
"""LangGraph node factories. make_nodes closes over an injected retriever and
chat model, so production wires real ones and tests wire fakes."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph_rag.state import AgentState

_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "Is the following document relevant to the question? Answer only 'yes' or 'no'.\n\n"
    "Question: {question}\n\nDocument: {document}"
)
_GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""
)
_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the question to improve document retrieval. Return only the rewritten "
    "question.\n\nQuestion: {question}"
)


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_path', '?')}#chunk{d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
        for d in docs
    )


def make_nodes(retriever, llm, max_attempts: int = 2) -> dict:
    grade_chain = _GRADE_PROMPT | llm | StrOutputParser()
    gen_chain = _GEN_PROMPT | llm | StrOutputParser()
    rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()

    def retrieve(state: AgentState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "attempts": state.get("attempts", 0) + 1}

    def grade_documents(state: AgentState) -> dict:
        kept = []
        for d in state["documents"]:
            verdict = grade_chain.invoke(
                {"question": state["question"], "document": d.page_content}
            )
            if verdict.strip().lower().startswith("y"):
                kept.append(d)
        return {"documents": kept}

    def generate(state: AgentState) -> dict:
        context = _format_docs(state["documents"])
        text = gen_chain.invoke({"context": context, "question": state["question"]})
        return {"generation": text}

    def transform_query(state: AgentState) -> dict:
        better = rewrite_chain.invoke({"question": state["question"]})
        return {"question": better.strip()}

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "generate": generate,
        "transform_query": transform_query,
    }


def make_decider(max_attempts: int = 2):
    def decide_to_generate(state: AgentState) -> str:
        if state["documents"]:
            return "generate"
        if state.get("attempts", 0) >= max_attempts:
            return "generate"
        return "transform_query"

    return decide_to_generate
```

- [ ] **Step 6: Create `graph.py`**

```python
"""Assemble the CRAG-lite agent graph."""

import os

from langgraph.graph import END, START, StateGraph

from langgraph_rag.nodes import make_decider, make_nodes
from langgraph_rag.state import AgentState


def build_graph(retriever, llm, max_attempts: int = 2):
    nodes = make_nodes(retriever, llm, max_attempts=max_attempts)
    decide = make_decider(max_attempts=max_attempts)

    g = StateGraph(AgentState)
    g.add_node("retrieve", nodes["retrieve"])
    g.add_node("grade_documents", nodes["grade_documents"])
    g.add_node("generate", nodes["generate"])
    g.add_node("transform_query", nodes["transform_query"])

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    g.add_edge("transform_query", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def build_agent(store, embedder, top_k: int = 3, max_attempts: int = 2):
    # Imported here so the module loads without langchain_rag/ollama at import time.
    from langchain_ollama import ChatOllama

    from langchain_rag.retriever import HybridRetriever

    retriever = HybridRetriever(store=store, embedder=embedder, top_k=top_k)
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    return build_graph(retriever, llm, max_attempts=max_attempts)
```

- [ ] **Step 7: Write the compile test**

Create `langgraph_rag/tests/test_graph.py`:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from langgraph_rag.graph import build_graph


def test_graph_compiles_with_expected_nodes():
    dummy_retriever = RunnableLambda(lambda q: [])
    llm = FakeListChatModel(responses=["yes"])
    graph = build_graph(dummy_retriever, llm)
    node_names = set(graph.get_graph().nodes)
    assert {"retrieve", "grade_documents", "generate", "transform_query"} <= node_names
```

- [ ] **Step 8: Run it**

Run: `.venv/bin/pytest langgraph_rag/tests/test_graph.py -v`
Expected: 1 test PASS (no infra).

- [ ] **Step 9: Confirm whole-suite collects**

Run: `.venv/bin/pytest --collect-only -q 2>&1 | tail -3`
Expected: collects with no errors.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml langgraph_rag/
git commit -m "feat: LangGraph CRAG-lite agent (state, nodes, graph)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Control-flow tests + demo subcommand

**Files:**
- Test: `langgraph_rag/tests/test_agent_flow.py`
- Modify: `demo.py` (add `agent` subcommand)

**Interfaces:**
- Consumes: `build_graph` (Task 1); `FakeListChatModel`, `RunnableLambda`, `Document`.
- Produces: `demo.py agent "<query>"`.

- [ ] **Step 1: Write the control-flow tests**

Create `langgraph_rag/tests/test_agent_flow.py`:

```python
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from langgraph_rag.graph import build_graph

_DOC = Document(
    page_content="Machine learning learns patterns from data.",
    metadata={"source_path": "zz.md", "chunk_index": 0},
)


def _fixed_retriever():
    # Returns exactly one doc per call -> one grade LLM call per round.
    return RunnableLambda(lambda q: [_DOC])


def _run(graph, question):
    return graph.invoke(
        {"question": question, "documents": [], "generation": "", "attempts": 0}
    )


def test_happy_path_grade_yes_then_generate():
    llm = FakeListChatModel(responses=["yes", "ML learns from data."])
    result = _run(build_graph(_fixed_retriever(), llm), "what is machine learning")
    assert result["generation"] == "ML learns from data."
    assert result["attempts"] == 1


def test_retry_loop_grade_no_then_transform_then_generate():
    # round 1: grade "no" -> transform -> round 2: grade "yes" -> generate
    llm = FakeListChatModel(
        responses=["no", "rewritten question", "yes", "Looped answer."]
    )
    result = _run(build_graph(_fixed_retriever(), llm), "vague query")
    assert result["attempts"] == 2
    assert result["generation"] == "Looped answer."
```

- [ ] **Step 2: Run to verify they fail then pass**

Run: `.venv/bin/pytest langgraph_rag/tests/test_agent_flow.py -v`
Expected: PASS (2 tests) once Task 1 is in place — these are the real deliverable of the capstone (deterministic agentic control flow). If `attempts` or `generation` mismatches, the graph wiring is wrong — fix the graph, not the test.

- [ ] **Step 3: Add the `agent` demo subcommand to `demo.py`**

In `demo.py`, insert this branch inside `main()` immediately after the existing `if selected == "langchain": ... return` block:

```python
    if selected == "agent":
        from langgraph_rag.graph import build_agent

        if len(sys.argv) < 3:
            print('Usage: python demo.py agent "<query>"')
            return
        query = sys.argv[2]
        with VectorStore() as store:
            graph = build_agent(store, Embedder())
            result = graph.invoke(
                {"question": query, "documents": [], "generation": "", "attempts": 0}
            )
        print(f"\nQuestion: {query}\nAnswer:   {result['generation']}")
        return
```

- [ ] **Step 4: Manually verify the demo (needs Ollama)**

```bash
docker compose up -d
.venv/bin/python demo.py ingest data/corpus
.venv/bin/python demo.py agent "what is machine learning"
```

Expected: a grounded answer (the agent retrieves, grades, and generates). Capture it in the report.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass, no collection errors.

- [ ] **Step 6: Commit**

```bash
git add langgraph_rag/tests/test_agent_flow.py demo.py
git commit -m "feat: LangGraph agent control-flow tests + demo subcommand

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Docs + final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Update `README.md`**

1. In the use-case table, add:

```markdown
| `python demo.py agent "<query>"` | Agentic RAG via LangGraph (retrieve → grade → rewrite-and-retry → generate) |
```

2. Under **Stack**, add:

```markdown
- **LangGraph**: a CRAG-lite agentic RAG graph (retrieve → grade documents → corrective retry → generate) over the same retrieval
```

3. In **Project Structure**, add:

```markdown
langgraph_rag/
  state.py       # AgentState (TypedDict)
  nodes.py       # retrieve / grade_documents / generate / transform_query
  graph.py       # build_graph (CRAG-lite StateGraph) + build_agent
```

4. Add an **Agentic RAG (LangGraph)** section before **Testing**:

````markdown
## Agentic RAG (LangGraph)

A LangGraph agent adds a corrective loop on top of retrieval: it retrieves,
LLM-grades each document's relevance, and if nothing relevant is found it
rewrites the query and retries (bounded), before generating.

```bash
pip install -e ".[langchain]"
python demo.py agent "what is machine learning"
```

The graph (`retrieve → grade_documents → {generate | transform_query → retrieve}`)
is built with injected retriever + chat model, so its control flow is unit-tested
deterministically with fakes — no Ollama needed for the tests.
````

- [ ] **Step 2: Final verification**

```bash
.venv/bin/pytest -q
```

Expected: full suite passes, no collection errors.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document LangGraph agentic RAG

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
