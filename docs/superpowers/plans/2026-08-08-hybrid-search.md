# Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Postgres full-text keyword retrieval alongside the existing pgvector search and fuse the two rankings with Reciprocal Rank Fusion, so the RAG chatbot retrieves on both meaning and exact terms.

**Architecture:** A new `search/` package holds a pure `reciprocal_rank_fusion` function (`fusion.py`) and a `hybrid_search` orchestrator (`hybrid.py`). `core/db.py` gains a `search_chunks_keyword` method (tsvector); the existing `search_chunks` (vector) is untouched. `rag_chatbot.retrieve()` switches to hybrid while `retrieve_vector()` preserves the vector-only path for a new `demo.py compare` subcommand.

**Tech Stack:** Python 3.11+, psycopg2 + pgvector (raw SQL, no ORM), Postgres full-text search (`tsvector`, `websearch_to_tsquery`, `ts_rank`, GIN index), pytest.

**Spec:** `docs/superpowers/specs/2026-08-07-hybrid-search-design.md`

## Global Constraints

- No LangChain/LlamaIndex — raw Python only. No BM25 extension; Postgres FTS only.
- RRF: `k=60` default, rank is **0-based**, output sorted by score descending with deterministic tie-break by ascending id.
- Keyword SQL uses `websearch_to_tsquery('english', ...)` (never plain `to_tsquery`) so raw user input can't raise.
- `search_chunks_keyword` returns the **same dict shape** as `search_chunks`: keys `id`, `content`, `chunk_index`, `source_path`, `score`.
- `candidate_k=20` fetched per engine; `top_k=3` into the prompt.
- The grounding prompt in `use_cases/rag_chatbot.py` must stay byte-identical (the promptfoo eval mirrors it).
- `content_tsv` is `GENERATED ALWAYS AS (to_tsvector('english', content)) STORED` with a GIN index.
- Existing `search_chunks` and the documents/products/logs code stay untouched.
- Integration tests: marked `@pytest.mark.integration`, use the deterministic `FakeEmbedder` (no Ollama), `zz-test-` path prefix, purge via `_purge_test_rows` before AND after each test.
- Run from repo root with the venv: `.venv/bin/pytest`, `.venv/bin/python`. Postgres via `docker compose up -d postgres`.
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Schema — content_tsv column + GIN index

**Files:**
- Modify: `init.sql`

**Interfaces:**
- Produces: `chunks.content_tsv` (tsvector, generated) and `chunks_content_tsv_idx` (GIN) in Postgres. Task 3 queries these.

- [ ] **Step 1: Append the column and index to `init.sql`**

Add at the end of `init.sql`:

```sql
ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING GIN (content_tsv);
```

- [ ] **Step 2: Apply to the running dev DB**

```bash
docker compose up -d postgres
docker compose exec -T postgres psql -U vectordb -d vectordb -f /docker-entrypoint-initdb.d/init.sql
```

Expected: `ALTER TABLE` and `CREATE INDEX` succeed (existing objects report `already exists, skipping`). The generated column backfills existing chunks automatically. (If `.env` overrides `POSTGRES_USER`/`POSTGRES_DB`, substitute those.)

- [ ] **Step 3: Verify the column and index exist and are populated**

```bash
docker compose exec -T postgres psql -U vectordb -d vectordb -c "\d chunks"
docker compose exec -T postgres psql -U vectordb -d vectordb -c "SELECT count(*) FROM chunks WHERE content_tsv IS NOT NULL;"
```

Expected: `\d chunks` shows `content_tsv | tsvector` (generated) and `chunks_content_tsv_idx` gin; the count matches the number of ingested chunks (backfill worked).

- [ ] **Step 4: Commit**

```bash
git add init.sql
git commit -m "feat: add generated content_tsv column and GIN index to chunks

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Reciprocal Rank Fusion (pure function)

**Files:**
- Create: `search/__init__.py` (empty)
- Create: `search/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Produces: `reciprocal_rank_fusion(ranked_id_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]` in `search.fusion`. Task 4 consumes it.

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p search && touch search/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_fusion.py`:

```python
from search.fusion import reciprocal_rank_fusion


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([[3, 1, 2]])
    assert [doc_id for doc_id, _ in fused] == [3, 1, 2]


def test_single_list_order_independent_of_k():
    order_small = [i for i, _ in reciprocal_rank_fusion([[3, 1, 2]], k=1)]
    order_large = [i for i, _ in reciprocal_rank_fusion([[3, 1, 2]], k=1000)]
    assert order_small == order_large == [3, 1, 2]


def test_id_in_both_lists_outranks_id_in_one():
    # doc 2 appears in both lists; doc 1 only in the first at rank 0.
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 5]])
    order = [doc_id for doc_id, _ in fused]
    assert order[0] == 2            # in both -> highest fused score
    assert order.index(1) < order.index(4)  # rank-0 single beats rank-1 single


def test_all_ids_present_as_union():
    fused = reciprocal_rank_fusion([[1, 2], [2, 3]])
    assert {doc_id for doc_id, _ in fused} == {1, 2, 3}


def test_score_formula_zero_based_rank():
    # single doc at rank 0 -> 1/(k+0)
    fused = reciprocal_rank_fusion([[7]], k=60)
    assert fused == [(7, 1.0 / 60)]


def test_tie_break_is_ascending_id():
    # doc 1 and doc 2 each appear once at rank 0 -> equal score -> id order
    fused = reciprocal_rank_fusion([[1], [2]])
    assert [doc_id for doc_id, _ in fused] == [1, 2]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'search.fusion'`.

- [ ] **Step 4: Implement the function**

Create `search/fusion.py`:

```python
"""Reciprocal Rank Fusion — merge several ranked ID lists into one ranking.

RRF is scale-free: it uses each item's rank position, not its raw score, so
rankings from different engines (cosine similarity, ts_rank) combine without
normalization. Reference: Cormack et al., TREC 2009.
"""


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists. Each ID scores Σ 1/(k + rank) over the lists it
    appears in (rank is 0-based). Returns (id, score) sorted by score
    descending, breaking ties by ascending id (deterministic)."""
    scores: dict[int, float] = {}
    for id_list in ranked_id_lists:
        for rank, doc_id in enumerate(id_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_fusion.py -v`
Expected: 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add search/__init__.py search/fusion.py tests/test_fusion.py
git commit -m "feat: reciprocal rank fusion pure function with unit tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Keyword search method

**Files:**
- Modify: `core/db.py` (append one method to `VectorStore`)
- Test: `tests/test_db_keyword.py`

**Interfaces:**
- Consumes: `_purge_test_rows` and `fake_embedding` from `tests.helpers` (existing); `VectorStore.upsert_source_with_chunks(root, path, chunks)` (existing).
- Produces: `VectorStore.search_chunks_keyword(query_text: str, top_k: int = 20) -> list[dict]` with dict keys `id`, `content`, `chunk_index`, `source_path`, `score`. Task 4 consumes it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db_keyword.py`:

```python
import pytest

from tests.helpers import _purge_test_rows, fake_embedding

pytestmark = pytest.mark.integration

ROOT = "zz-test-kw"
PATH = "zz-test-kw/doc.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def _insert(store, content):
    store.upsert_source_with_chunks(ROOT, PATH, [(content, 3, fake_embedding(content))])


def test_keyword_finds_exact_term(store):
    _insert(store, "Photosynthesis converts sunlight into glucose.")
    rows = store.search_chunks_keyword("photosynthesis", top_k=20)
    assert any(r["source_path"] == PATH for r in rows)


def test_keyword_no_match_returns_empty(store):
    _insert(store, "Photosynthesis converts sunlight into glucose.")
    rows = store.search_chunks_keyword("zqxjkbrstvwx", top_k=20)
    assert rows == []


def test_keyword_garbage_input_does_not_raise(store):
    # websearch_to_tsquery must swallow operator garbage instead of erroring.
    assert store.search_chunks_keyword('"unterminated AND OR -', top_k=20) == []


def test_keyword_result_shape_matches_vector(store):
    _insert(store, "Photosynthesis converts sunlight into glucose.")
    rows = store.search_chunks_keyword("photosynthesis", top_k=20)
    ours = [r for r in rows if r["source_path"] == PATH]
    assert set(ours[0]) == {"id", "content", "chunk_index", "source_path", "score"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_db_keyword.py -v`
Expected: FAIL — `AttributeError: 'VectorStore' object has no attribute 'search_chunks_keyword'` (start Postgres first: `docker compose up -d postgres`).

- [ ] **Step 3: Implement the method**

Append to the `VectorStore` class in `core/db.py`, immediately after `search_chunks`:

```python
    def search_chunks_keyword(self, query_text: str, top_k: int = 20) -> list[dict]:
        """Full-text keyword search over chunks. Uses websearch_to_tsquery so
        raw user input never raises; a query with no lexical matches returns
        []. Same result shape as search_chunks."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id, c.content, c.chunk_index, s.path AS source_path,
                       ts_rank(c.content_tsv, websearch_to_tsquery('english', %s)) AS score
                FROM chunks c
                JOIN sources s ON s.id = c.source_id
                WHERE c.content_tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_text, query_text, top_k),
            )
            return [dict(r) for r in cur.fetchall()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_db_keyword.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/db.py tests/test_db_keyword.py
git commit -m "feat: full-text keyword search over chunks (tsvector)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Hybrid orchestration + RAG integration

**Files:**
- Create: `search/hybrid.py`
- Modify: `use_cases/rag_chatbot.py` (retrieve → hybrid; add retrieve_vector)
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: `reciprocal_rank_fusion` (Task 2); `VectorStore.search_chunks` (existing) and `search_chunks_keyword` (Task 3); `ChunkResult` from `core.models` (fields: `id`, `content`, `source_path`, `chunk_index`, `score`); `FakeEmbedder`/`fake_embedding`/`_purge_test_rows` from `tests.helpers`.
- Produces: `hybrid_search(query: str, store, embedder, top_k: int = 3, candidate_k: int = 20) -> list[ChunkResult]` in `search.hybrid`; `retrieve_vector(query, store, embedder, top_k=3) -> list[ChunkResult]` in `use_cases.rag_chatbot`; `retrieve` now delegates to `hybrid_search`. Task 5 consumes `retrieve` and `retrieve_vector`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hybrid.py`:

```python
import pytest

from search.hybrid import hybrid_search
from tests.helpers import FakeEmbedder, _purge_test_rows, fake_embedding

pytestmark = pytest.mark.integration

ROOT = "zz-test-hy"


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def _insert(store, path, content):
    store.upsert_source_with_chunks(
        ROOT, path, [(content, 3, fake_embedding(content))]
    )


def test_returns_chunk_results(store):
    _insert(store, "zz-test-hy/a.md", "The frobnicator handles edge cases.")
    results = hybrid_search("frobnicator", store, FakeEmbedder(), top_k=3)
    assert results and all(hasattr(r, "source_path") for r in results)


def test_keyword_distinctive_term_is_retrieved(store):
    # A term FakeEmbedder's vector won't surface, but keyword search will.
    _insert(store, "zz-test-hy/kw.md", "Xylophone zebra quokka distinctive term.")
    results = hybrid_search("quokka", store, FakeEmbedder(), top_k=5)
    assert any(r.source_path == "zz-test-hy/kw.md" for r in results)


def test_chunk_in_both_engines_is_deduped(store):
    _insert(store, "zz-test-hy/dup.md", "photosynthesis photosynthesis photosynthesis")
    results = hybrid_search("photosynthesis", store, FakeEmbedder(), top_k=20)
    paths = [r.source_path for r in results]
    assert paths.count("zz-test-hy/dup.md") == 1


def test_empty_query_returns_empty(store):
    assert hybrid_search("   ", store, FakeEmbedder(), top_k=3) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'search.hybrid'`.

- [ ] **Step 3: Implement the orchestrator**

Create `search/hybrid.py`:

```python
"""Hybrid retrieval: fuse vector and keyword search with RRF.

Runs vector (cosine) and keyword (tsvector) search independently, each
returning up to candidate_k ranked chunks, then merges their rankings with
reciprocal rank fusion and returns the top_k as ChunkResult objects. The
returned score is the RRF score.
"""

from core.db import VectorStore
from core.embedder import Embedder
from core.models import ChunkResult
from search.fusion import reciprocal_rank_fusion


def hybrid_search(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int = 3,
    candidate_k: int = 20,
) -> list[ChunkResult]:
    if not query.strip():
        return []

    embedding = embedder.embed(query)
    vec_rows = store.search_chunks(embedding, top_k=candidate_k)
    kw_rows = store.search_chunks_keyword(query, top_k=candidate_k)

    rows_by_id = {r["id"]: r for r in vec_rows + kw_rows}  # union, dedup by id
    fused = reciprocal_rank_fusion(
        [[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]]
    )
    return [
        ChunkResult(**{**rows_by_id[doc_id], "score": score})
        for doc_id, score in fused[:top_k]
    ]
```

- [ ] **Step 4: Run the hybrid tests to verify they pass**

Run: `.venv/bin/pytest tests/test_hybrid.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Wire hybrid into the RAG chatbot**

In `use_cases/rag_chatbot.py`, add the import after the existing `from ingestion.ingest import ingest_directory` line:

```python
from search.hybrid import hybrid_search
```

Then replace the existing `retrieve` function:

```python
def retrieve(
    query: str, store: VectorStore, embedder: Embedder, top_k: int = 3
) -> list[ChunkResult]:
    embedding = embedder.embed(query)
    rows = store.search_chunks(embedding, top_k=top_k)
    return [ChunkResult(**r) for r in rows]
```

with both of these (vector-only path preserved, hybrid becomes the default):

```python
def retrieve_vector(
    query: str, store: VectorStore, embedder: Embedder, top_k: int = 3
) -> list[ChunkResult]:
    """Vector-only retrieval — kept for the `demo.py compare` subcommand."""
    embedding = embedder.embed(query)
    rows = store.search_chunks(embedding, top_k=top_k)
    return [ChunkResult(**r) for r in rows]


def retrieve(
    query: str, store: VectorStore, embedder: Embedder, top_k: int = 3
) -> list[ChunkResult]:
    return hybrid_search(query, store, embedder, top_k=top_k)
```

(`generate_answer`, `chat`, `run`, and the grounding prompt are unchanged.)

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest`
Expected: all tests pass (fusion unit + keyword/hybrid/db/ingest/chunker/loader). Needs Postgres up for the integration tests.

- [ ] **Step 7: Commit**

```bash
git add search/hybrid.py use_cases/rag_chatbot.py tests/test_hybrid.py
git commit -m "feat: hybrid search orchestration; RAG retrieves hybrid, keeps vector-only

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Compare demo + docs + final verification

**Files:**
- Modify: `demo.py` (add `compare` subcommand)
- Modify: `README.md`

**Interfaces:**
- Consumes: `retrieve` (hybrid) and `retrieve_vector` from `use_cases.rag_chatbot` (Task 4); `VectorStore`, `Embedder`.

- [ ] **Step 1: Add the `compare` subcommand to `demo.py`**

In `demo.py`, insert this branch inside `main()` immediately after the existing `if selected == "ingest": ... return` block:

```python
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
```

- [ ] **Step 2: Manually verify the compare demo**

```bash
docker compose up -d
.venv/bin/python demo.py ingest data/corpus
.venv/bin/python demo.py compare "what are vector databases for"
```

Expected: a two-column table printing vector-only vs hybrid rankings for the query (5 rows, `source#chunk` labels). Both columns populated; no traceback. Capture the output for the report.

- [ ] **Step 3: Update `README.md`**

1. In the use-case table, add this row directly below the `ingest` row:

```markdown
| `python demo.py compare "<query>"` | Compare vector-only vs hybrid (RRF) retrieval rankings side by side |
```

2. Under **Stack**, add:

```markdown
- **Hybrid search**: pgvector cosine + Postgres full-text (`tsvector`), fused with Reciprocal Rank Fusion (k=60)
```

3. In **Project Structure**, add after the `ingestion/` block:

```markdown
search/
  fusion.py      # reciprocal rank fusion (pure function)
  hybrid.py      # vector + keyword retrieval fused into one ranking
```

4. In **Key Concepts**, replace the RAG Chatbot paragraph with:

```markdown
**RAG Chatbot**: ingests `data/corpus`, then retrieves with **hybrid search** — pgvector semantic search and Postgres keyword search fused by Reciprocal Rank Fusion — and passes the top chunks with source citations to Ollama. `python demo.py compare "<query>"` shows the vector-only vs hybrid rankings side by side.
```

- [ ] **Step 4: Final verification**

```bash
.venv/bin/pytest
.venv/bin/python demo.py compare "what are vector databases for"
```

Expected: all tests pass; the compare table prints cleanly.

- [ ] **Step 5: Commit**

```bash
git add demo.py README.md
git commit -m "feat: add compare demo and document hybrid search

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
