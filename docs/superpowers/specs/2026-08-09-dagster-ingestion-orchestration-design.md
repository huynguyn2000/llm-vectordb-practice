# Dagster Ingestion Orchestration — Design

**Date:** 2026-08-09
**Status:** Approved
**Context:** First sub-project of the DE-layer milestone ("show RAG sits on top
of real data-engineering work"). Orchestrates the existing idempotent
file-ingestion pipeline (`ingest_directory`) as Dagster **assets** run locally
via `dagster dev`. No cloud, no Docker. A later sub-project (DuckDB + dbt) will
reuse this Dagster foundation to orchestrate a structured-data source.

## Milestone decomposition

Full DE story: **DuckDB (structured source) → dbt (transform to a clean
documents table) → Dagster (orchestrate: run dbt, extract to text, ingest+embed
into pgvector)**. Too large for one spec, so split into two sub-projects, each
working on its own:

- **Sub-project A (this spec):** Dagster orchestrates the existing file
  ingestion. No warehouse. Stands up the orchestration foundation.
- **Sub-project B (later):** DuckDB + dbt as a structured document source,
  orchestrated by A's Dagster. DuckDB is fully local and $0 (chosen over
  BigQuery/Redshift for zero cost and higher learning value).

## Why Dagster (not Airflow)

The user already runs Airflow professionally; Dagster is the larger learning
step and a cleaner fit for this local stack. Dagster's **asset-centric** model
(declare data assets and their dependencies, not imperative tasks) maps
naturally onto the RAG pipeline's lineage (`corpus files → chunks/embeddings in
pgvector`) and integrates first-class with dbt and DuckDB in sub-project B. It
runs as a local Python process (`dagster dev`) — no Docker, far lighter than
Airflow.

## Goals

- Run the existing `ingest_directory` pipeline as Dagster assets, on a schedule
  and manually, via `dagster dev`.
- Reuse the tested project code unchanged — assets call `ingest_directory`
  directly through injected resources.
- Demonstrate asset lineage, an asset check, resource-based dependency
  injection, materialization metadata, and scheduling.
- Zero cost; no changes to `core/`, `ingestion/`, or `search/`.

## Non-goals

- DuckDB, dbt, or any warehouse integration (sub-project B).
- Production Dagster deployment (Dagster+, Kubernetes, Docker executor).
- Sensors/event-driven runs — a daily schedule plus manual materialization is
  enough.
- Changing the ingestion pipeline itself.

## Architecture & layout

Dagster code lives in a new `orchestration/` package. `dagster dev` runs in the
project's virtualenv, so it imports the real project packages directly — no
Docker image, no bind-mount.

```
orchestration/
  __init__.py
  resources.py       # Dagster resources wrapping VectorStore + Embedder (DI)
  assets.py          # corpus_source, pgvector_chunks assets + chunks_present check
  definitions.py     # Definitions(assets=[...], asset_checks=[...], schedules=[...], resources={...})
  tests/
    test_definitions.py   # definitions load + structure, no infra
pyproject.toml       # + dagster, dagster-webserver (dev deps); [tool.dagster] module pointer
```

**Zero-code-change integration & networking:** because `dagster dev` runs on
the host in the same venv, assets reach pgvector Postgres and Ollama on plain
`localhost` using the existing env defaults in `core/db.py` and
`core/embedder.py`. No `host.docker.internal`, no port remap, no container
networking. The corpus is read from the local `data/corpus` path directly.

## Resources (`orchestration/resources.py`)

Two Dagster resources inject the data dependencies, keeping assets testable:

- `vector_store` — yields a `VectorStore` (opened per run, closed on teardown).
- `embedder` — provides an `Embedder`.

In tests, the `embedder` resource is swapped for a `FakeEmbedder` (from
`tests.helpers`) so assets can be exercised without Ollama.

## Assets & check (`orchestration/assets.py`)

The pipeline is declared as assets with real lineage:

```python
@asset
def corpus_source() -> MaterializeResult:
    # Count supported files (.md/.txt/.pdf) in data/corpus.
    # Raise if zero (fail fast). Emit file count as metadata.

@asset(deps=[corpus_source])
def pgvector_chunks(vector_store, embedder) -> MaterializeResult:
    # ingest_directory(CORPUS_DIR, vector_store, embedder)
    # Return MaterializeResult with IngestStats (ingested/skipped/deleted/failed)
    # attached as Dagster metadata.

@asset_check(asset=pgvector_chunks)
def chunks_present(vector_store) -> AssetCheckResult:
    # SELECT count(*) FROM chunks; passed = count > 0; report count as metadata.
```

- **Lineage:** `corpus_source → pgvector_chunks` shows the data dependency the
  UI renders as a graph.
- **Asset check** `chunks_present` is Dagster's idiomatic post-condition (the
  "verify" step), separate from the materialization.
- **Metadata** on materializations surfaces `IngestStats` and chunk counts in
  the Dagster UI run history.

## Definitions & schedule (`orchestration/definitions.py`)

- A `Definitions` object registers the assets, the asset check, the resources,
  a job that materializes the assets, and a daily `ScheduleDefinition`.
- Manual materialization is available one-click in the Dagster UI.
- Idempotent hash-diff ingestion keeps daily re-runs cheap (unchanged files
  skipped).
- `pyproject.toml` gains `[tool.dagster] module_name = "orchestration.definitions"`
  so `dagster dev` finds the code location.

## Error handling

- Postgres/Ollama unreachable → the asset raises a clear connection error; the
  run is marked failed in the UI. No silent success.
- Empty/absent corpus → `corpus_source` raises before any embedding work.
- Infra errors propagate as run failures (consistent with the ingestion
  pipeline's fail-fast-on-infra stance); data problems inside `ingest_directory`
  are counted in `IngestStats.failed` and surfaced via materialization metadata.
- The `chunks_present` check turns "materialization didn't error" into "data is
  actually present."

## Testing

- `orchestration/tests/test_definitions.py` (pytest, **no infra**): the
  `Definitions` object loads (catches import/wiring errors) and exposes the
  expected assets (`corpus_source`, `pgvector_chunks`), the `chunks_present`
  check, and the daily schedule. Analogous to an Airflow DAG-integrity test.
- Integration test (real Postgres + `FakeEmbedder` resource): materialize
  `pgvector_chunks` and assert the check passes and `IngestStats` is populated.
  Uses the existing `zz-test-` conventions and purge fixtures.
- Existing project test suite is untouched — this sub-project adds no code to
  `core/`, `ingestion/`, or `search/`.

## Cost & resources

| | Money | Main resource |
|---|---|---|
| Sub-project A (this spec) | **$0** — local Python process, Dagster OSS free | Laptop RAM |

- **RAM:** `dagster dev` is one lightweight Python process (webserver + daemon),
  a few hundred MB — much lighter than Airflow's container stack. The real
  footprint stays Ollama (`llama3.2` + `nomic-embed-text` ≈ 5–8 GB) + pgvector.
- **Disk:** Dagster's local instance storage (SQLite under `DAGSTER_HOME`) is
  tiny.
- **Cloud:** none. (Sub-project B adds DuckDB + dbt, both free/local — ~$0,
  documented in that spec.)

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Orchestrator | Dagster (`dagster dev`) | Asset-centric paradigm (new vs the user's Airflow experience); first-class dbt/DuckDB fit; $0 local; lighter than Airflow |
| Execution model | Assets call `ingest_directory` via injected resources | Reuses tested code; idiomatic Dagster DI; testable |
| Pipeline shape | `corpus_source → pgvector_chunks` + `chunks_present` check | Shows lineage, asset checks, and metadata — the data-aware patterns |
| Schedule | Daily `ScheduleDefinition` + manual materialization | Idempotent re-runs are safe/cheap |
| Networking/config | Runs on host; `localhost` + existing env defaults | No Docker/port-remap needed; zero code change |
| Code location | `orchestration/` package + `[tool.dagster]` pointer | Contained, discoverable by `dagster dev` |
| Warehouse (sub-project B) | DuckDB + dbt-duckdb | $0 forever, local, high learning value vs BigQuery/Redshift |
