# Vector-DB Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Abstract vector retrieval behind a `VectorBackend` protocol, add a ChromaDB backend beside pgvector, and compare them on the same corpus.

**Architecture:** New `vectorstores/` package: `base.py` (protocol), `pgvector.py` (wraps `VectorStore.search_chunks`), `chroma.py` (embedded chromadb), `corpus.py` (populate Chroma from `data/corpus`). A `demo.py vsdb` compares top-k side by side. Additive — the main RAG path is untouched; chromadb is an opt-in `chroma` extra with lazy import.

**Tech Stack:** Python 3.11+, chromadb (opt-in extra), the existing pgvector + Ollama + ingestion stack, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-vector-db-comparison-design.md`

## Global Constraints

- `VectorBackend` protocol = `name: str` + `search(embedding, top_k=5) -> list[dict]` with keys `id`, `content`, `source_path`, `chunk_index`, `score` (cosine similarity, higher=better) — identical to `VectorStore.search_chunks`. (Population/`upsert` is backend-specific, not in the protocol.)
- `PgvectorBackend(store).search` delegates to `store.search_chunks`. `ChromaBackend` uses embedded chromadb (cosine space), converts Chroma distance → `score = 1 - distance`.
- chromadb imported **lazily** (inside methods/classmethods) so `vectorstores` imports without it. `chromadb>=0.5` in a `chroma` extra, NOT base.
- Reuse `ingestion.loaders`/`ingestion.chunker`/`core.embedder` to populate Chroma. No changes to core/search/ingestion/use_cases (additive package only).
- Chroma tests use in-memory `EphemeralClient` + `tests.helpers.fake_embedding`; `pytest.importorskip("chromadb")` so they skip cleanly if chromadb absent.
- Add `vectorstores` (+ `vectorstores.tests`) to `[tool.setuptools] packages`.
- Every commit ends with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Backends package + corpus builder + tests

**Files:**
- Modify: `pyproject.toml` (chroma extra + packages)
- Create: `vectorstores/__init__.py` (empty), `base.py`, `pgvector.py`, `chroma.py`, `corpus.py`
- Create: `vectorstores/tests/__init__.py` (empty)
- Test: `vectorstores/tests/test_chroma.py`, `vectorstores/tests/test_pgvector.py`

**Interfaces:**
- Produces: `VectorBackend` (base), `PgvectorBackend(store)` (pgvector), `ChromaBackend` with `.in_memory()`/`.persistent()` classmethods + `upsert`/`search` (chroma), `build_chroma_backend_from_corpus(corpus_dir, embedder, backend=None) -> ChromaBackend` (corpus). Task 2 consumes `PgvectorBackend`, `build_chroma_backend_from_corpus`.

- [ ] **Step 1: Add chroma extra + packages to `pyproject.toml`**

```toml
chroma = ["chromadb>=0.5"]
```
Add `vectorstores`, `vectorstores.tests` to `[tool.setuptools] packages`.

- [ ] **Step 2: Install (watch for dep conflicts early)**

```bash
pip install -e ".[dev,chroma]"
```
Then IMMEDIATELY run `.venv/bin/pytest --collect-only -q` to confirm chromadb didn't break collection of any existing test (the Docling milestone hit an antlr/dagster conflict this way). If collection breaks, report the conflict (which package/version) as a BLOCKER rather than pressing on.

- [ ] **Step 3: `vectorstores/base.py`**

```python
"""Vector backend protocol — the search interface both stores share."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    name: str

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return up to top_k results, each a dict with keys id, content,
        source_path, chunk_index, score (cosine similarity, higher = better)."""
        ...
```

- [ ] **Step 4: `vectorstores/pgvector.py`**

```python
"""pgvector-backed VectorBackend (wraps the existing VectorStore)."""

from core.db import VectorStore


class PgvectorBackend:
    name = "pgvector"

    def __init__(self, store: VectorStore):
        self._store = store

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        return self._store.search_chunks(embedding, top_k=top_k)
```

- [ ] **Step 5: `vectorstores/chroma.py`**

```python
"""ChromaDB-backed VectorBackend (embedded; cosine space). chromadb imported
lazily so this module loads without the optional extra."""


class ChromaBackend:
    name = "chroma"

    def __init__(self, collection):
        self._collection = collection

    @classmethod
    def _collection_for(cls, client, name: str):
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    @classmethod
    def in_memory(cls, name: str = "chunks") -> "ChromaBackend":
        import chromadb

        return cls(cls._collection_for(chromadb.EphemeralClient(), name))

    @classmethod
    def persistent(cls, path: str = ".chroma", name: str = "chunks") -> "ChromaBackend":
        import chromadb

        return cls(cls._collection_for(chromadb.PersistentClient(path=path), name))

    def upsert(self, items: list[dict]) -> None:
        self._collection.upsert(
            ids=[str(it["id"]) for it in items],
            documents=[it["content"] for it in items],
            embeddings=[it["embedding"] for it in items],
            metadatas=[
                {"source_path": it["source_path"], "chunk_index": it["chunk_index"]}
                for it in items
            ],
        )

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]
        out = []
        for i in range(len(ids)):
            raw_id = ids[i]
            out.append(
                {
                    "id": int(raw_id) if raw_id.isdigit() else raw_id,
                    "content": docs[i],
                    "source_path": metas[i].get("source_path"),
                    "chunk_index": metas[i].get("chunk_index"),
                    "score": 1.0 - dists[i],
                }
            )
        return out
```

- [ ] **Step 6: `vectorstores/corpus.py`**

```python
"""Populate a Chroma backend from a corpus directory, reusing the ingestion
loaders + chunker so it holds the same chunks as pgvector."""

from pathlib import Path

from ingestion.chunker import chunk_text
from ingestion.loaders import SUPPORTED_EXTENSIONS, load_file
from vectorstores.chroma import ChromaBackend


def build_chroma_backend_from_corpus(corpus_dir, embedder, backend=None) -> ChromaBackend:
    backend = backend or ChromaBackend.persistent()
    root = Path(corpus_dir)
    items, cid = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        doc = load_file(path)
        rel = str(path.relative_to(root))
        for ch in chunk_text(doc.text):
            items.append(
                {
                    "id": cid,
                    "content": ch.content,
                    "source_path": rel,
                    "chunk_index": ch.chunk_index,
                    "embedding": embedder.embed(ch.content),
                }
            )
            cid += 1
    if items:
        backend.upsert(items)
    return backend
```

- [ ] **Step 7: `vectorstores/tests/test_chroma.py` (no-infra)**

```python
import pytest

pytest.importorskip("chromadb")

from tests.helpers import fake_embedding
from vectorstores.chroma import ChromaBackend


def test_upsert_and_search_ranks_closest_first():
    backend = ChromaBackend.in_memory()
    items = [
        {"id": 1, "content": "alpha", "source_path": "a.md", "chunk_index": 0, "embedding": fake_embedding("alpha")},
        {"id": 2, "content": "beta", "source_path": "b.md", "chunk_index": 1, "embedding": fake_embedding("beta")},
        {"id": 3, "content": "gamma", "source_path": "c.md", "chunk_index": 2, "embedding": fake_embedding("gamma")},
    ]
    backend.upsert(items)
    res = backend.search(fake_embedding("beta"), top_k=3)
    assert res, "expected results"
    assert res[0]["source_path"] == "b.md"  # query vector == beta's vector -> ranks first
    assert set(res[0]) == {"id", "content", "source_path", "chunk_index", "score"}
    assert -1e-6 <= res[0]["score"] <= 1.0 + 1e-6


def test_search_result_count_capped_by_top_k():
    backend = ChromaBackend.in_memory()
    backend.upsert(
        [{"id": i, "content": f"c{i}", "source_path": "s.md", "chunk_index": i, "embedding": fake_embedding(f"c{i}")} for i in range(5)]
    )
    assert len(backend.search(fake_embedding("c0"), top_k=2)) == 2
```

- [ ] **Step 8: `vectorstores/tests/test_pgvector.py` (integration)**

```python
import pytest

from tests.helpers import _purge_test_rows, fake_embedding
from vectorstores.pgvector import PgvectorBackend

pytestmark = pytest.mark.integration

ROOT = "zz-test-vdb"
PATH = "zz-test-vdb/a.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def test_pgvector_backend_search_shape(store):
    store.upsert_source_with_chunks(
        ROOT, PATH, "hash", [("alpha content", 3, fake_embedding("alpha content"))]
    )
    backend = PgvectorBackend(store)
    assert backend.name == "pgvector"
    res = backend.search(fake_embedding("alpha content"), top_k=5)
    ours = [r for r in res if r["source_path"] == PATH]
    assert ours
    assert set(ours[0]) == {"id", "content", "source_path", "chunk_index", "score"}
```

(`vectorstores/tests/` needs the `store` fixture; add `vectorstores/tests/conftest.py` with `from tests.conftest import store  # noqa: F401` — the re-export pattern, NOT pytest_plugins.)

- [ ] **Step 9: Run tests**

```bash
docker compose up -d postgres
.venv/bin/pytest vectorstores/tests/ -v
```
Expected: chroma tests pass (chromadb installed via the extra), pgvector integration test passes. Then `.venv/bin/pytest -q` full suite — confirm no regressions/collection errors.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml vectorstores/
git commit -m "feat: VectorBackend abstraction with pgvector + ChromaDB backends

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Compare demo + docs

**Files:**
- Modify: `demo.py` (add `vsdb` subcommand)
- Modify: `README.md`

**Interfaces:**
- Consumes: `PgvectorBackend`, `build_chroma_backend_from_corpus` (Task 1); `VectorStore`, `Embedder`.

- [ ] **Step 1: Add the `vsdb` subcommand to `demo.py`**

Insert in `main()` after the existing `if selected == "compare": ... return` block:

```python
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
```

- [ ] **Step 2: Manual verify (needs Postgres + Ollama + chroma extra + corpus ingested)**

```bash
docker compose up -d
.venv/bin/python demo.py ingest data/corpus
.venv/bin/python demo.py vsdb "what are vector databases for"
```
Expected: a `rank | pgvector | chroma` table (both columns populated with source#chunk + score). Capture it. If chromadb install/first-build is heavy or hits a dep conflict, document it (same honest posture as prior milestones) — the no-infra chroma unit tests remain the correctness gate.

- [ ] **Step 3: Update `README.md`**

1. Use-case table row: `| \`python demo.py vsdb "<query>"\` | Compare pgvector vs ChromaDB vector search side by side |`
2. Stack bullet: `- **Vector-store abstraction**: a \`VectorBackend\` protocol with pgvector and ChromaDB (embedded) backends; \`demo.py vsdb\` compares them (Weaviate is a documented next adapter)`
3. Project Structure: add the `vectorstores/` block (base/pgvector/chroma/corpus).
4. A short **Vector-DB comparison** section: install `.[chroma]`, run `demo.py vsdb`, and a note that Weaviate would slot in as another `VectorBackend` (needs a Docker service, not built here).

- [ ] **Step 4: Final verification**

```bash
.venv/bin/pytest -q
```
Expected: full suite passes (chroma tests run if chromadb installed, else skip), no collection errors.

- [ ] **Step 5: Commit**

```bash
git add demo.py README.md
git commit -m "feat: vsdb compare demo + document vector-store abstraction

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
