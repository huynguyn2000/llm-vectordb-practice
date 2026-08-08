# LangChain RAG Pipeline + LangSmith Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangChain LCEL RAG chain that reuses the project's hybrid retrieval via a custom retriever, with env-gated LangSmith tracing, running fully local on Ollama.

**Architecture:** New `langchain_rag/` package: `retriever.py` (a `BaseRetriever` wrapping `search.hybrid.hybrid_search`) and `chain.py` (an LCEL chain: retriever → grounding prompt → `ChatOllama` → parser). LangSmith activates from env vars only. A `demo.py langchain` subcommand invokes it.

**Tech Stack:** Python 3.11+, LangChain (`langchain`, `langchain-core`, `langchain-ollama`, `langsmith`), the existing pgvector + Ollama + hybrid-search stack, pytest.

**Spec:** `docs/superpowers/specs/2026-08-09-langchain-rag-design.md`

## Global Constraints

- Local only: LLM via `langchain-ollama` `ChatOllama` (`LLM_MODEL`/`OLLAMA_BASE_URL` env, default `llama3.2` / `http://localhost:11434`, `temperature=0`). No cloud LLM, no API key to run.
- Reuse existing retrieval: the retriever calls `hybrid_search(query, store, embedder, top_k=top_k)` and maps `ChunkResult` → `Document`. No re-ingestion, no new table, no changes to `core/`, `ingestion/`, `search/`.
- `ChunkResult` fields: `id`, `content`, `source_path`, `chunk_index`, `score`.
- LangSmith: env-gated (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`), OFF by default; no code beyond run-name/tags on the chain.
- Citation format in `format_docs`: `[Source: {source_path}#chunk{chunk_index}]\n{content}` (matches the raw pipeline).
- Grounding prompt mirrors the existing one: answer only from context; otherwise exactly `I don't know based on the provided documents.`
- Integration tests: `@pytest.mark.integration`, `FakeEmbedder`, `zz-test-` prefix, purge before AND after. **`langchain_rag/tests/conftest.py` must re-export the `store` fixture via `from tests.conftest import store  # noqa: F401` — NOT `pytest_plugins` (that breaks whole-suite collection on pytest 9).**
- Run from repo root with the venv: `.venv/bin/pytest`, `.venv/bin/python`. Install with `pip install -e ".[dev,langchain]"`.
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Deps + custom retriever + retriever test

**Files:**
- Modify: `pyproject.toml` (langchain extra + packages)
- Create: `langchain_rag/__init__.py` (empty)
- Create: `langchain_rag/retriever.py`
- Create: `langchain_rag/tests/__init__.py` (empty)
- Create: `langchain_rag/tests/conftest.py`
- Test: `langchain_rag/tests/test_retriever.py`

**Interfaces:**
- Produces: `HybridRetriever(BaseRetriever)` with fields `store: VectorStore`, `embedder: Embedder`, `top_k: int = 3` in `langchain_rag.retriever`, returning `langchain_core.documents.Document` objects. Task 2 consumes `HybridRetriever`.

- [ ] **Step 1: Add the langchain extra and package to `pyproject.toml`**

Under `[project.optional-dependencies]` add:

```toml
langchain = ["langchain>=0.3,<0.4", "langchain-ollama>=0.2", "langsmith>=0.1"]
```

Add `langchain_rag` to the packages list:

```toml
[tool.setuptools]
packages = ["core", "use_cases", "data", "ingestion", "search", "orchestration", "langchain_rag"]
```

(If `orchestration` isn't present because that branch isn't merged, still add both `orchestration` is not required — include at least `langchain_rag`; keep any packages already listed.)

- [ ] **Step 2: Install**

```bash
pip install -e ".[dev,langchain]"
```

Expected: langchain, langchain-core, langchain-ollama, langsmith install.

- [ ] **Step 3: Create package markers**

```bash
mkdir -p langchain_rag/tests && touch langchain_rag/__init__.py langchain_rag/tests/__init__.py
```

- [ ] **Step 4: Create the fixture bridge**

Create `langchain_rag/tests/conftest.py`:

```python
"""Re-export the shared `store` fixture into this sibling test dir. Importing a
fixture into a conftest's namespace shares it without the deprecated
`pytest_plugins` mechanism (which errors in a non-root conftest on pytest 9)."""

from tests.conftest import store  # noqa: F401
```

- [ ] **Step 5: Write the failing retriever test**

Create `langchain_rag/tests/test_retriever.py`:

```python
import pytest
from langchain_core.documents import Document

from langchain_rag.retriever import HybridRetriever
from tests.helpers import FakeEmbedder, _purge_test_rows, fake_embedding

pytestmark = pytest.mark.integration

ROOT = "zz-test-lc"
PATH = "zz-test-lc/doc.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def test_retriever_returns_documents(store):
    store.upsert_source_with_chunks(
        ROOT, PATH, "test-hash", [("Photosynthesis converts sunlight.", 3, fake_embedding("x"))]
    )
    retriever = HybridRetriever(store=store, embedder=FakeEmbedder(), top_k=3)
    docs = retriever.invoke("photosynthesis")
    assert docs and all(isinstance(d, Document) for d in docs)
    ours = [d for d in docs if d.metadata.get("source_path") == PATH]
    assert ours, "expected the ingested chunk among results"
    assert set(ours[0].metadata) >= {"source_path", "chunk_index", "score", "id"}
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/bin/pytest langchain_rag/tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'langchain_rag.retriever'` (start Postgres first: `docker compose up -d postgres`).

- [ ] **Step 7: Implement the retriever**

Create `langchain_rag/retriever.py`:

```python
"""A LangChain retriever that reuses the project's hybrid search.

Wrapping the existing hybrid_search (vector + keyword + RRF) keeps a single
source of retrieval truth and avoids a second embeddings table."""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from core.db import VectorStore
from core.embedder import Embedder
from search.hybrid import hybrid_search


class HybridRetriever(BaseRetriever):
    """Retrieve chunks via hybrid_search, as LangChain Documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: VectorStore
    embedder: Embedder
    top_k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = hybrid_search(query, self.store, self.embedder, top_k=self.top_k)
        return [
            Document(
                page_content=r.content,
                metadata={
                    "source_path": r.source_path,
                    "chunk_index": r.chunk_index,
                    "score": r.score,
                    "id": r.id,
                },
            )
            for r in results
        ]
```

- [ ] **Step 8: Run it to verify it passes**

Run: `.venv/bin/pytest langchain_rag/tests/test_retriever.py -v`
Expected: 1 test PASS (not skipped).

- [ ] **Step 9: Confirm whole-suite still collects**

Run: `.venv/bin/pytest --collect-only -q 2>&1 | tail -3`
Expected: collects with no errors (verifies the conftest bridge didn't break collection).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml langchain_rag/__init__.py langchain_rag/retriever.py langchain_rag/tests/
git commit -m "feat: LangChain HybridRetriever wrapping hybrid search

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: LCEL RAG chain + demo subcommand

**Files:**
- Create: `langchain_rag/chain.py`
- Test: `langchain_rag/tests/test_chain.py`
- Modify: `demo.py` (add `langchain` subcommand)

**Interfaces:**
- Consumes: `HybridRetriever` (Task 1); `VectorStore`, `Embedder`.
- Produces: `format_docs(docs: list[Document]) -> str` and `build_rag_chain(store, embedder, top_k=3) -> Runnable` in `langchain_rag.chain`. Task 3 (docs) references the demo.

- [ ] **Step 1: Write the failing tests**

Create `langchain_rag/tests/test_chain.py`:

```python
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from langchain_rag.chain import build_rag_chain, format_docs
from tests.helpers import FakeEmbedder, _purge_test_rows, fake_embedding


def test_format_docs_citation():
    docs = [
        Document(page_content="Alpha.", metadata={"source_path": "a.md", "chunk_index": 0}),
        Document(page_content="Beta.", metadata={"source_path": "b.md", "chunk_index": 2}),
    ]
    assert format_docs(docs) == (
        "[Source: a.md#chunk0]\nAlpha.\n\n[Source: b.md#chunk2]\nBeta."
    )


@pytest.mark.integration
def test_build_rag_chain_is_runnable(store):
    # Constructing the chain opens a DB connection (VectorStore) but does NOT
    # call the LLM, so no Ollama is needed here.
    _purge_test_rows(store)
    chain = build_rag_chain(store, FakeEmbedder(), top_k=3)
    assert isinstance(chain, Runnable)
    assert hasattr(chain, "invoke")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest langchain_rag/tests/test_chain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'langchain_rag.chain'`.

- [ ] **Step 3: Implement the chain**

Create `langchain_rag/chain.py`:

```python
"""LangChain LCEL RAG chain over the project's hybrid retrieval, generating with
a local Ollama model. LangSmith tracing activates from env vars (off by default);
the chain carries a run name + tags so traces are legible."""

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama import ChatOllama

from core.db import VectorStore
from core.embedder import Embedder
from langchain_rag.retriever import HybridRetriever

_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['source_path']}#chunk{d.metadata['chunk_index']}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(store: VectorStore, embedder: Embedder, top_k: int = 3) -> Runnable:
    retriever = HybridRetriever(store=store, embedder=embedder, top_k=top_k)
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.with_config(run_name="langchain_rag", tags=["rag", "hybrid"])
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest langchain_rag/tests/test_chain.py -v`
Expected: 2 tests PASS (Postgres up for the integration one; no Ollama needed).

- [ ] **Step 5: Add the `langchain` demo subcommand to `demo.py`**

In `demo.py`, insert this branch inside `main()` immediately after the existing `if selected == "compare": ... return` block:

```python
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
```

- [ ] **Step 6: Manually verify the demo (needs Ollama)**

```bash
docker compose up -d
.venv/bin/python demo.py ingest data/corpus
.venv/bin/python demo.py langchain "what is machine learning"
```

Expected: a grounded answer sourced from the corpus. Capture it in the report.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass, no collection errors.

- [ ] **Step 8: Commit**

```bash
git add langchain_rag/chain.py langchain_rag/tests/test_chain.py demo.py
git commit -m "feat: LangChain LCEL RAG chain + demo subcommand

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: LangSmith wiring + docs + final verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Add LangSmith env vars to `.env.example`**

Append:

```bash
# LangSmith tracing (optional; off unless you set these). Get a key at smith.langchain.com
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls-...
# LANGCHAIN_PROJECT=llm-vectordb-practice
```

- [ ] **Step 2: Update `README.md`**

1. In the use-case table, add:

```markdown
| `python demo.py langchain "<query>"` | RAG answer via the LangChain LCEL chain (same hybrid retrieval, local Ollama) |
```

2. Under **Stack**, add:

```markdown
- **LangChain**: an LCEL RAG chain (`langchain-ollama`) over the same hybrid retrieval, with optional LangSmith tracing
```

3. In **Project Structure**, add:

```markdown
langchain_rag/
  retriever.py   # BaseRetriever wrapping hybrid_search -> LangChain Documents
  chain.py       # LCEL RAG chain (retriever -> prompt -> ChatOllama)
```

4. Add a **LangChain & LangSmith** section before **Testing**:

````markdown
## LangChain & LangSmith

A LangChain LCEL variant of the RAG pipeline reuses the same hybrid retrieval:

```bash
pip install -e ".[langchain]"
python demo.py langchain "what is machine learning"
```

To trace runs in **LangSmith**, set the env vars in `.env` (a free key from
smith.langchain.com) — tracing is off by default:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=llm-vectordb-practice
```

The chain is tagged (`run_name="langchain_rag"`, tags `rag`/`hybrid`) so traces
are legible in the LangSmith UI.
````

- [ ] **Step 3: Final verification**

```bash
.venv/bin/pytest -q
```

Expected: full suite passes, no collection errors.

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md
git commit -m "docs: document LangChain RAG chain and LangSmith tracing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
