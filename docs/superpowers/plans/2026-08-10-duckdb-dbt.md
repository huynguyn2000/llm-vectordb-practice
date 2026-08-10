# DuckDB + dbt Structured Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A DuckDB warehouse + dbt transform that turns raw structured records into a `documents` table, exported as a text corpus the existing RAG ingestion consumes — demonstrating RAG on top of real DE work.

**Architecture:** `de/dbt/` is a dbt-duckdb project (seed → staging → `documents` mart). `de/pipeline.py` runs dbt, reads the `documents` table from DuckDB, and writes each row to `data/warehouse_corpus/`. `demo.py warehouse` runs the flow. Opt-in `de` extra; DuckDB is in-process (no Docker).

**Tech Stack:** Python 3.11+, dbt-duckdb + duckdb (opt-in `de` extra), the existing ingestion pipeline, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-duckdb-dbt-design.md`

## Global Constraints

- `de` extra = `["dbt-duckdb>=1.8", "duckdb>=1.0"]`, NOT base. Add `de`, `de.tests` to setuptools packages.
- DuckDB is in-process; dbt runs via the `dbt` CLI as a subprocess with `cwd=de/dbt`, `--project-dir .`, `--profiles-dir .`.
- `documents` mart columns: `id`, `source` (`'faq:' || id`), `content` (question + blank line + answer).
- `de/pipeline.py`: `run_dbt(project_dir="de/dbt")`, `export_documents(duckdb_path) -> list[dict]{id,source,content}`, `write_corpus(rows, out_dir="data/warehouse_corpus") -> int`. Missing `dbt`/`duckdb` → clear error naming the `de` extra.
- Generated artifacts gitignored: `de/dbt/warehouse.duckdb`, `de/dbt/target/`, `de/dbt/logs/`, `data/warehouse_corpus/`.
- Reuse the existing ingestion for the RAG side; no changes to core/search/ingestion/use_cases.
- Tests: `test_export.py` needs only `duckdb` (no dbt); `test_dbt_build.py` needs the `de` extra (skip cleanly if `dbt`/`duckdb` absent).
- Every commit ends with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: dbt project + pipeline + tests

**Files:**
- Modify: `pyproject.toml` (de extra + packages), `.gitignore`
- Create: `de/__init__.py` (empty), `de/pipeline.py`
- Create: `de/dbt/dbt_project.yml`, `de/dbt/profiles.yml`, `de/dbt/seeds/faq.csv`, `de/dbt/models/staging/stg_faq.sql`, `de/dbt/models/marts/documents.sql`
- Create: `de/tests/__init__.py` (empty), `test_export.py`, `test_dbt_build.py`

**Interfaces:**
- Produces: `run_dbt`, `export_documents`, `write_corpus`, `DUCKDB_PATH` (de.pipeline). Task 2 consumes them.

- [ ] **Step 1: pyproject + gitignore**

Add under optional-dependencies: `de = ["dbt-duckdb>=1.8", "duckdb>=1.0"]`. Add `de`, `de.tests` to `[tool.setuptools] packages`. Append to `.gitignore`:
```
de/dbt/warehouse.duckdb
de/dbt/target/
de/dbt/logs/
data/warehouse_corpus/
```

- [ ] **Step 2: Install + collection check**

```bash
pip install -e ".[dev,de]"
.venv/bin/pytest --collect-only -q
```
dbt-core pins click/jinja2/agate — if the install breaks collection of any existing test (a Docling-style conflict), STOP and report the conflict; otherwise proceed.

- [ ] **Step 3: package markers** — `mkdir -p de/dbt/seeds de/dbt/models/staging de/dbt/models/marts de/tests && touch de/__init__.py de/tests/__init__.py`

- [ ] **Step 4: `de/dbt/dbt_project.yml`**

```yaml
name: "warehouse"
version: "1.0.0"
profile: "warehouse"
model-paths: ["models"]
seed-paths: ["seeds"]
target-path: "target"
clean-targets: ["target", "dbt_packages"]
models:
  warehouse:
    staging:
      +materialized: view
    marts:
      +materialized: table
```

- [ ] **Step 5: `de/dbt/profiles.yml`**

```yaml
warehouse:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: warehouse.duckdb
      schema: main
      threads: 1
```

- [ ] **Step 6: `de/dbt/seeds/faq.csv`**

```csv
id,question,answer,topic
1,What is a vector database?,A vector database indexes high-dimensional embeddings so nearest-neighbor similarity search stays fast across millions of items.,vectors
2,What is an embedding?,An embedding is a dense numeric vector that represents the meaning of text so similar text has nearby vectors.,vectors
3,What is chunking in RAG?,Chunking splits documents into smaller passages so retrieval returns focused, relevant context to the model.,rag
4,What is hybrid search?,Hybrid search combines vector similarity with keyword matching and fuses the rankings for better recall.,rag
5,What is pgvector?,pgvector is a PostgreSQL extension that stores vectors and runs similarity search with indexes like HNSW.,postgres
6,What is dbt?,dbt is a transformation tool that lets you build, test, and document SQL models in your data warehouse.,de
7,What is DuckDB?,DuckDB is an in-process analytical (OLAP) database, like SQLite for analytics, great for local pipelines.,de
8,What is retrieval-augmented generation?,RAG retrieves relevant documents and passes them as context to an LLM so answers are grounded in your data.,rag
```

- [ ] **Step 7: `de/dbt/models/staging/stg_faq.sql`**

```sql
select
    id,
    trim(question) as question,
    trim(answer) as answer,
    lower(topic) as topic
from {{ ref('faq') }}
```

- [ ] **Step 8: `de/dbt/models/marts/documents.sql`**

```sql
select
    id,
    'faq:' || id as source,
    question || chr(10) || chr(10) || answer as content
from {{ ref('stg_faq') }}
```

- [ ] **Step 9: `de/pipeline.py`**

```python
"""DuckDB + dbt structured-source pipeline: build the warehouse with dbt, then
export the `documents` mart as a text corpus for the RAG ingestion."""

import shutil
import subprocess
from pathlib import Path

DBT_DIR = Path("de/dbt")
DUCKDB_PATH = DBT_DIR / "warehouse.duckdb"


def run_dbt(project_dir=DBT_DIR) -> None:
    """Run `dbt build` in the dbt project. Raises if dbt isn't installed or fails."""
    if shutil.which("dbt") is None:
        raise RuntimeError(
            "dbt CLI not found — install the de extra: pip install -e '.[de]'"
        )
    result = subprocess.run(
        ["dbt", "build", "--project-dir", ".", "--profiles-dir", "."],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed:\n{result.stdout}\n{result.stderr}")


def export_documents(duckdb_path=DUCKDB_PATH) -> list[dict]:
    """Read the dbt `documents` mart from DuckDB."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb not installed — install the de extra: pip install -e '.[de]'"
        ) from exc
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT id, source, content FROM documents ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    return [{"id": r[0], "source": r[1], "content": r[2]} for r in rows]


def write_corpus(rows, out_dir="data/warehouse_corpus") -> int:
    """Write each document row to a .md file the RAG ingestion can consume."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        slug = str(row["source"]).replace(":", "_").replace("/", "_")
        (out / f"{slug}.md").write_text(row["content"], encoding="utf-8")
    return len(rows)
```

- [ ] **Step 10: `de/tests/test_export.py` (no dbt)**

```python
import pytest

duckdb = pytest.importorskip("duckdb")

from de.pipeline import export_documents, write_corpus


def test_export_and_write(tmp_path):
    dbpath = tmp_path / "w.duckdb"
    con = duckdb.connect(str(dbpath))
    con.execute("CREATE TABLE documents (id INTEGER, source VARCHAR, content VARCHAR)")
    con.execute(
        "INSERT INTO documents VALUES (1, 'faq:1', 'Q one\n\nA one'), (2, 'faq:2', 'Q two\n\nA two')"
    )
    con.close()

    rows = export_documents(dbpath)
    assert len(rows) == 2
    assert rows[0] == {"id": 1, "source": "faq:1", "content": "Q one\n\nA one"}

    n = write_corpus(rows, out_dir=str(tmp_path / "corpus"))
    assert n == 2
    assert (tmp_path / "corpus" / "faq_1.md").read_text() == "Q one\n\nA one"
```

- [ ] **Step 11: `de/tests/test_dbt_build.py` (de extra)**

```python
import shutil

import pytest

pytest.importorskip("duckdb")
if shutil.which("dbt") is None:
    pytest.skip("dbt CLI not installed (de extra)", allow_module_level=True)

from de.pipeline import export_documents, run_dbt


def test_dbt_build_produces_documents():
    run_dbt()
    rows = export_documents()
    assert len(rows) >= 8  # one per seed row
    assert all(r["content"].strip() for r in rows)
    assert all(str(r["source"]).startswith("faq:") for r in rows)
    # content is question + answer text
    assert any("vector database" in r["content"].lower() for r in rows)
```

- [ ] **Step 12: Run tests**

```bash
.venv/bin/pytest de/tests/ -v
```
Expected: `test_export` passes (needs only duckdb); `test_dbt_build` runs a real `dbt build` on the seed and passes (DuckDB in-process, fast). Then `.venv/bin/pytest -q` full suite — no regressions/collection errors.

- [ ] **Step 13: Commit**

```bash
git add pyproject.toml .gitignore de/
git commit -m "feat: DuckDB + dbt structured source (seed -> documents mart -> corpus)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: warehouse demo + docs

**Files:**
- Modify: `demo.py` (add `warehouse` subcommand)
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_dbt`, `export_documents`, `write_corpus` (Task 1).

- [ ] **Step 1: Add the `warehouse` subcommand to `demo.py`** (after existing subcommand branches, before USE_CASES validation):

```python
    if selected == "warehouse":
        try:
            from de.pipeline import export_documents, run_dbt, write_corpus
        except ImportError as exc:
            print(exc)
            return
        try:
            run_dbt()
            rows = export_documents()
        except (ImportError, RuntimeError) as exc:
            print(exc)
            return
        n = write_corpus(rows)
        print(f"Built {n} documents from the DuckDB/dbt warehouse -> data/warehouse_corpus/")
        print("Next: python demo.py ingest data/warehouse_corpus")
        return
```

- [ ] **Step 2: Manual verify (needs the de extra; no Postgres/Ollama for the build itself)**

```bash
pip install -e ".[de]"
.venv/bin/python demo.py warehouse
ls data/warehouse_corpus/
```
Expected: prints "Built 8 documents … -> data/warehouse_corpus/" and the dir has 8 `.md` files. Capture the output. (Optionally, with Postgres+Ollama up: `python demo.py ingest data/warehouse_corpus` to push them through RAG — capture if you run it, but it's not required.)

- [ ] **Step 3: Update `README.md`**

1. Use-case row: `| \`python demo.py warehouse\` | Build a DuckDB warehouse with dbt and export its \`documents\` mart to a RAG corpus |`
2. Stack bullet: `- **Structured source (DE)**: a DuckDB warehouse transformed by dbt (seed → staging → \`documents\` mart), exported to a corpus the RAG pipeline ingests — RAG on top of real data-engineering work`
3. Project Structure: `de/` block (dbt/ project + pipeline.py).
4. A **Structured source (DuckDB + dbt)** section: `pip install -e ".[de]"`, `demo.py warehouse`, then `demo.py ingest data/warehouse_corpus`; explain the seed → staging → documents flow and note that orchestrating dbt from Dagster (`dagster-dbt`) is the natural integration (not wired here).

- [ ] **Step 4: Final verification**

```bash
.venv/bin/pytest -q
```
Expected: full suite passes, no collection errors.

- [ ] **Step 5: Commit**

```bash
git add demo.py README.md
git commit -m "feat: warehouse demo + document DuckDB/dbt structured source

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
