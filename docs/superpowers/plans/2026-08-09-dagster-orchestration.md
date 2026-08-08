# Dagster Ingestion Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orchestrate the existing idempotent file-ingestion pipeline as Dagster assets (with lineage, an asset check, resources, and a daily schedule) run locally via `dagster dev`.

**Architecture:** A new `orchestration/` package declares two assets (`corpus_source → pgvector_chunks`) plus an asset check (`chunks_present`), with `VectorStore`/`Embedder` injected via Dagster resources. `dagster dev` runs in the project venv, so assets import the real `ingestion`/`core` code and reach pgvector + Ollama on `localhost` with zero code change to those packages.

**Tech Stack:** Python 3.11+, Dagster (`dagster` + `dagster-webserver`, 1.8.x), the existing pgvector + Ollama stack, pytest.

**Spec:** `docs/superpowers/specs/2026-08-09-dagster-ingestion-orchestration-design.md`

## Global Constraints

- Dagster pinned `>=1.8,<2`; `dagster-webserver` same range.
- No changes to `core/`, `ingestion/`, or `search/`. The chunk-count check reads `store.conn` directly (a public attribute); no new `VectorStore` method.
- Assets resolve the corpus directory from `os.getenv("CORPUS_DIR", "data/corpus")` **at runtime** (inside each asset), so integration tests can point at a temp dir without touching the real corpus.
- `dagster dev` runs on the host (not Docker); assets reach Postgres/Ollama on `localhost` via the existing env defaults in `core/db.py` / `core/embedder.py` — no `host.docker.internal`, no port remap.
- Resources: `VectorStoreResource.get_store() -> VectorStore` and `EmbedderResource.get_embedder() -> Embedder`; the caller closes the store.
- Daily schedule cron `"0 6 * * *"`; job name `"ingest_corpus_job"`; schedule name `"daily_ingest"`.
- Integration tests: marked `@pytest.mark.integration`, use a `FakeEmbedder` resource (no Ollama), point `CORPUS_DIR` at a temp dir containing `zz-test-`-prefixed files, and purge via `_purge_test_rows` before AND after.
- Run from repo root with the venv: `.venv/bin/pytest`, `.venv/bin/dagster`. Install with `pip install -e ".[dev,orchestration]"`.
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Dagster scaffolding — deps, resources, assets, definitions, load test

**Files:**
- Modify: `pyproject.toml` (deps, packages, `[tool.dagster]`)
- Create: `orchestration/__init__.py` (empty)
- Create: `orchestration/resources.py`
- Create: `orchestration/assets.py`
- Create: `orchestration/definitions.py`
- Create: `orchestration/tests/__init__.py` (empty)
- Test: `orchestration/tests/test_definitions.py`

**Interfaces:**
- Produces: `VectorStoreResource` / `EmbedderResource` (in `orchestration.resources`); assets `corpus_source`, `pgvector_chunks` and check `chunks_present` (in `orchestration.assets`); `defs`, `ingest_job`, `daily_schedule` (in `orchestration.definitions`). Task 2 consumes the assets, the check, and `VectorStoreResource`.

- [ ] **Step 1: Add dependencies and Dagster config to `pyproject.toml`**

Add an orchestration extra under `[project.optional-dependencies]` (alongside the existing `dev`):

```toml
orchestration = ["dagster>=1.8,<2", "dagster-webserver>=1.8,<2"]
```

Add `orchestration` to the packages list:

```toml
[tool.setuptools]
packages = ["core", "use_cases", "data", "ingestion", "search", "orchestration"]
```

Add a Dagster code-location pointer at the end of the file:

```toml
[tool.dagster]
module_name = "orchestration.definitions"
```

- [ ] **Step 2: Install**

```bash
pip install -e ".[dev,orchestration]"
```

Expected: dagster + dagster-webserver install successfully.

- [ ] **Step 3: Create the package markers**

```bash
mkdir -p orchestration/tests && touch orchestration/__init__.py orchestration/tests/__init__.py
```

- [ ] **Step 4: Write the resources**

Create `orchestration/resources.py`:

```python
"""Dagster resources wrapping the project's data dependencies, so assets stay
testable (swap EmbedderResource for a fake in tests). These construct clients
lazily in their getters — instantiating the resource opens no connections."""

from dagster import ConfigurableResource

from core.db import VectorStore
from core.embedder import Embedder


class VectorStoreResource(ConfigurableResource):
    """Provides a pgvector VectorStore. The caller is responsible for closing it."""

    def get_store(self) -> VectorStore:
        return VectorStore()


class EmbedderResource(ConfigurableResource):
    """Provides an Ollama Embedder."""

    def get_embedder(self) -> Embedder:
        return Embedder()
```

- [ ] **Step 5: Write the assets and check**

Create `orchestration/assets.py`:

```python
"""The ingestion pipeline declared as Dagster assets with lineage
(corpus_source -> pgvector_chunks) plus a post-condition asset check.

The corpus directory is read from the CORPUS_DIR env var at runtime (default
data/corpus) so tests can retarget it without touching the real corpus."""

import os
from pathlib import Path

from dagster import (
    AssetCheckResult,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from ingestion.ingest import ingest_directory
from ingestion.loaders import SUPPORTED_EXTENSIONS
from orchestration.resources import EmbedderResource, VectorStoreResource


def _corpus_dir() -> str:
    return os.getenv("CORPUS_DIR", "data/corpus")


@asset
def corpus_source() -> MaterializeResult:
    """Validate the corpus directory has supported files; fail fast if empty."""
    root = Path(_corpus_dir())
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No supported files found in {root}")
    return MaterializeResult(
        metadata={
            "file_count": MetadataValue.int(len(files)),
            "corpus_dir": MetadataValue.text(str(root)),
        }
    )


@asset(deps=[corpus_source])
def pgvector_chunks(
    vector_store: VectorStoreResource, embedder: EmbedderResource
) -> MaterializeResult:
    """Ingest the corpus into pgvector (idempotent hash-diff). Emit IngestStats."""
    store = vector_store.get_store()
    try:
        stats = ingest_directory(_corpus_dir(), store, embedder.get_embedder())
    finally:
        store.close()
    return MaterializeResult(
        metadata={
            "ingested": MetadataValue.int(stats.ingested),
            "skipped": MetadataValue.int(stats.skipped),
            "deleted": MetadataValue.int(stats.deleted),
            "failed": MetadataValue.int(stats.failed),
        }
    )


@asset_check(asset=pgvector_chunks)
def chunks_present(vector_store: VectorStoreResource) -> AssetCheckResult:
    """Post-condition: pgvector holds at least one chunk."""
    store = vector_store.get_store()
    try:
        with store.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            count = cur.fetchone()[0]
    finally:
        store.close()
    return AssetCheckResult(
        passed=count > 0, metadata={"chunk_count": MetadataValue.int(count)}
    )
```

- [ ] **Step 6: Write the definitions**

Create `orchestration/definitions.py`:

```python
"""Dagster code location: assets, the asset check, a materialization job, a
daily schedule, and the resources they need."""

from dagster import Definitions, ScheduleDefinition, define_asset_job

from orchestration.assets import chunks_present, corpus_source, pgvector_chunks
from orchestration.resources import EmbedderResource, VectorStoreResource

ingest_job = define_asset_job(name="ingest_corpus_job", selection="*")

daily_schedule = ScheduleDefinition(
    name="daily_ingest",
    job=ingest_job,
    cron_schedule="0 6 * * *",
)

defs = Definitions(
    assets=[corpus_source, pgvector_chunks],
    asset_checks=[chunks_present],
    jobs=[ingest_job],
    schedules=[daily_schedule],
    resources={
        "vector_store": VectorStoreResource(),
        "embedder": EmbedderResource(),
    },
)
```

- [ ] **Step 7: Write the definitions-load test (no infra)**

Create `orchestration/tests/test_definitions.py`:

```python
"""Load/structure tests — no Postgres or Ollama needed. Constructing the
Definitions and resolving the job validates the asset graph and that every
resource key an asset/check requests is provided; resources are lazy, so no
connection is opened here."""

from orchestration import definitions as d
from orchestration.assets import corpus_source, pgvector_chunks


def test_definitions_import_ok():
    assert d.defs is not None


def test_job_resolves():
    # Resolving the job builds the asset graph and checks resource requirements
    # are satisfied — raises if wiring is broken.
    assert d.defs.get_job_def("ingest_corpus_job") is not None


def test_assets_have_expected_keys():
    assert corpus_source.key.path[-1] == "corpus_source"
    assert pgvector_chunks.key.path[-1] == "pgvector_chunks"


def test_schedule_is_daily():
    assert d.daily_schedule.cron_schedule == "0 6 * * *"
    assert d.daily_schedule.name == "daily_ingest"
```

- [ ] **Step 8: Run the load test**

Run: `.venv/bin/pytest orchestration/tests/test_definitions.py -v`
Expected: 4 tests PASS (no Postgres/Ollama needed).

- [ ] **Step 9: Confirm the code location loads in Dagster**

```bash
.venv/bin/dagster definitions validate
```

Expected: reports the `orchestration.definitions` location loaded successfully (assets `corpus_source`, `pgvector_chunks`; check `chunks_present`; schedule `daily_ingest`). This catches wiring errors `dagster dev` would hit.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml orchestration/
git commit -m "feat: Dagster code location — ingestion assets, check, schedule

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Integration test — materialize against real Postgres

**Files:**
- Test: `orchestration/tests/test_materialize.py`

**Interfaces:**
- Consumes: `corpus_source`, `pgvector_chunks`, `chunks_present` (Task 1); `VectorStoreResource` (Task 1); `FakeEmbedder`, `_purge_test_rows` from `tests.helpers`; the `store` fixture from `tests/conftest.py`.

- [ ] **Step 1: Write the integration test**

Create `orchestration/tests/test_materialize.py`:

```python
"""Materialize the assets + check against real Postgres with a FakeEmbedder
resource (no Ollama). CORPUS_DIR is pointed at a temp dir of zz-test files so
the real corpus's embeddings are never overwritten."""

import pytest
from dagster import ConfigurableResource, materialize

from orchestration.assets import chunks_present, corpus_source, pgvector_chunks
from orchestration.resources import VectorStoreResource
from tests.helpers import FakeEmbedder, _purge_test_rows

pytestmark = pytest.mark.integration


class FakeEmbedderResource(ConfigurableResource):
    def get_embedder(self):
        return FakeEmbedder()


@pytest.fixture
def temp_corpus(tmp_path, monkeypatch):
    (tmp_path / "zz-test-doc.md").write_text(
        "# Test\n\nDagster orchestration integration test document.",
        encoding="utf-8",
    )
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def cleanup(store):
    _purge_test_rows(store)
    yield
    _purge_test_rows(store)


def test_materialize_ingests_and_check_passes(store, temp_corpus):
    result = materialize(
        [corpus_source, pgvector_chunks, chunks_present],
        resources={
            "vector_store": VectorStoreResource(),
            "embedder": FakeEmbedderResource(),
        },
    )
    assert result.success

    evaluations = result.get_asset_check_evaluations()
    assert evaluations, "expected the chunks_present check to run"
    assert all(e.passed for e in evaluations)
```

- [ ] **Step 2: Run the integration test**

Run: `.venv/bin/pytest orchestration/tests/test_materialize.py -v`
Expected: 1 test PASS (start Postgres first: `docker compose up -d postgres`; no Ollama needed — FakeEmbedder). Not skipped.

- [ ] **Step 3: Manually verify in the Dagster UI**

```bash
docker compose up -d
.venv/bin/dagster dev
```

In the UI (http://localhost:3000): materialize `corpus_source` then `pgvector_chunks`, confirm both succeed with metadata (file_count; ingested/skipped/deleted/failed), and the `chunks_present` check shows passed with a `chunk_count`. Capture the outcome in the report, then Ctrl-C to stop.

- [ ] **Step 4: Commit**

```bash
git add orchestration/tests/test_materialize.py
git commit -m "test: integration materialize of Dagster assets against pgvector

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Documentation + final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Update `README.md`**

1. Under **Stack**, add:

```markdown
- **Orchestration**: Dagster — the ingestion pipeline as assets (`corpus_source → pgvector_chunks`) with an asset check and a daily schedule; run locally with `dagster dev`
```

2. In **Project Structure**, add after the `search/` block:

```markdown
orchestration/
  resources.py   # Dagster resources: VectorStore + Embedder (dependency injection)
  assets.py      # corpus_source, pgvector_chunks assets + chunks_present check
  definitions.py # Dagster code location: assets, check, job, daily schedule
```

3. Add an **Orchestration** section before **Testing**:

````markdown
## Orchestration (Dagster)

The file-ingestion pipeline is orchestrated by Dagster as assets. Run it locally:

```bash
docker compose up -d                 # Postgres (pgvector) + Ollama
pip install -e ".[orchestration]"    # dagster + dagster-webserver
dagster dev                          # UI at http://localhost:3000
```

In the UI, materialize `pgvector_chunks` (it depends on `corpus_source`) or wait
for the daily schedule. The `chunks_present` asset check verifies pgvector holds
at least one chunk after each run. Ingestion is idempotent (SHA-256 hash-diff),
so scheduled re-runs only re-embed changed files.
````

- [ ] **Step 2: Final verification**

```bash
.venv/bin/pytest
.venv/bin/dagster definitions validate
```

Expected: the full suite passes (existing tests + the new definitions-load and materialize tests; the materialize test needs Postgres up), and the Dagster code location validates.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document Dagster orchestration

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
