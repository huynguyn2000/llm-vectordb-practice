# Airflow Ingestion Orchestration — Design

**Date:** 2026-08-09
**Status:** Approved
**Context:** First sub-project of the DE-layer milestone ("show RAG sits on top
of real data-engineering work"). Wraps the existing idempotent file-ingestion
pipeline (`ingest_directory`) in an Airflow DAG run via the Astronomer CLI. No
GCP required. A later sub-project (BigQuery + dbt) will reuse this Airflow
foundation to orchestrate a structured-data source.

## Milestone decomposition

The full DE story is a pipeline: **BigQuery (structured source) → dbt
(transform to a clean documents table) → Airflow (orchestrate: run dbt, extract
to text, ingest+embed into pgvector)**. That is too large for one spec, so it is
split into two sub-projects, each producing working software on its own:

- **Sub-project A (this spec):** Airflow orchestrates the existing file
  ingestion. No GCP. Stands up the orchestration foundation.
- **Sub-project B (later):** BigQuery + dbt as a structured document source,
  orchestrated by A's Airflow, using a **personal** GCP project and a BigQuery
  **public dataset** (never the work `aha-move` project).

## Goals

- Run the existing `ingest_directory` pipeline on a schedule (and manually) via
  Airflow, using the Astronomer CLI for local Airflow.
- Reuse the tested project code unchanged — the DAG calls `ingest_directory`
  directly in a `PythonOperator`/`@task`.
- Demonstrate atomic task design (validate → ingest → verify) and scheduling.
- Zero cloud cost; no changes to `core/`, `ingestion/`, or `search/`.

## Non-goals

- BigQuery, dbt, or any GCP integration (sub-project B).
- Production Airflow deployment (Cloud Composer, Kubernetes executor, etc.).
- Sensors/event-driven triggers — a daily schedule plus manual trigger is enough.
- Changing the ingestion pipeline itself.

## Architecture & layout

The Astro project lives in a new `airflow/` subdirectory so `astro dev init`'s
files stay contained and do not scatter across the repo root.

```
airflow/                      # astro project (its own Docker build context)
  Dockerfile                  # FROM astro-runtime; installs runtime deps
  requirements.txt            # psycopg2-binary, pgvector, tiktoken, pypdf, ollama, pydantic, python-dotenv
  dags/
    ingest_corpus.py          # the DAG (3 tasks)
  tests/
    test_dag_integrity.py     # import + structure test, no infra
  .env                        # POSTGRES_HOST / OLLAMA_BASE_URL overrides (see below)
  airflow_settings.yaml       # local Airflow config incl. metadata-DB host-port remap
```

**Zero-code-change integration:** `core/db.py` and `core/embedder.py` already
read `POSTGRES_HOST`, `POSTGRES_PORT`, `OLLAMA_BASE_URL` (etc.) from the
environment with localhost defaults. Inside the Astro containers these are set
to `host.docker.internal`, so the existing code connects to the already-running
pgvector Postgres and Ollama with no modification.

**Package importability:** the repo root is bind-mounted into the Airflow
containers and added to `PYTHONPATH`, so `from ingestion.ingest import
ingest_directory` and `from core.db import VectorStore` resolve to the real,
tested packages. The corpus (`data/corpus`) is read through the same mount.

**Runtime dependencies** are installed into the Astro image via
`airflow/requirements.txt` (the project's runtime deps, minus Airflow itself,
which the base image provides).

## Networking

- The DAG reaches pgvector Postgres and Ollama via `host.docker.internal`
  (Mac): `POSTGRES_HOST=host.docker.internal`,
  `OLLAMA_BASE_URL=http://host.docker.internal:11434`, set in the Astro
  project's `.env`.
- **Port-clash resolution:** Astro's own metadata Postgres defaults to host port
  5432, colliding with the project's pgvector. Astro's metadata DB is remapped
  to host port **5433** (via `airflow_settings.yaml` / Astro config), leaving
  the project's pgvector on 5432 untouched.

## The DAG (`dags/ingest_corpus.py`)

TaskFlow API (`@task`), wired `validate_corpus → ingest → verify_chunk_count`:

```python
@task validate_corpus() -> int
    # Count supported files (.md/.txt/.pdf) in data/corpus.
    # Fail fast (raise) if zero — no point embedding nothing.

@task ingest(file_count: int) -> dict
    # with VectorStore() as store: ingest_directory(CORPUS_DIR, store, Embedder())
    # Return IngestStats as a dict via XCom.

@task verify_chunk_count(stats: dict) -> None
    # SELECT count(*) FROM chunks; assert > 0.
    # Log ingested / skipped / deleted / failed from stats.
```

**DAG config:**
- `schedule="@daily"`, `catchup=False`, plus manual UI trigger.
- `retries=1`, `retry_delay=timedelta(seconds=30)` — a transient
  "service not up yet" resolves on retry.
- Static `start_date` (no `datetime.now()` at parse time).
- `max_active_runs=1` — never two ingests racing.
- Each task opens and closes its own `VectorStore` (Airflow tasks are separate
  processes; connections do not survive across task boundaries).

**Why this shape:** `validate_corpus` is the fail-fast input guard; `ingest`
reuses the idempotent hash-diff pipeline so `@daily` re-runs are cheap
(unchanged files skipped); `verify_chunk_count` is the post-condition that turns
"task didn't error" into "the data is actually present."

## Error handling

- Postgres/Ollama unreachable → task raises a clear connection error; Airflow
  retries once, then marks the run failed in the UI. No silent success.
- Empty/absent corpus → `validate_corpus` raises before any embedding work.
- Infra errors propagate as task failures (consistent with the ingestion
  pipeline's fail-fast-on-infra stance); data problems inside `ingest_directory`
  are already counted in `IngestStats.failed` and surfaced by the verify task's
  logging.

## Testing

- `airflow/tests/test_dag_integrity.py` (pytest, **no infra**): imports all DAGs
  (catches syntax/import errors), asserts the `ingest_corpus` DAG exists with
  exactly three tasks and the `validate_corpus → ingest → verify_chunk_count`
  dependency order. Runs in CI without Postgres/Ollama.
- Manual verification: `astro dev start`, trigger the DAG from the UI, confirm
  all three tasks succeed and `verify_chunk_count` logs the `IngestStats`.
- Existing project test suite is untouched — this sub-project adds no code to
  `core/`, `ingestion/`, or `search/`.

## Cost & resources

| | Money | Main resource |
|---|---|---|
| Sub-project A (this spec) | **$0** — fully local, Astro CLI free | Laptop RAM |

- **RAM:** Astro runs ~4 containers (webserver, scheduler, triggerer, metadata
  Postgres) ≈ 2–4 GB, on top of Ollama (`llama3.2` + `nomic-embed-text` ≈ 5–8 GB)
  and pgvector Postgres. **All up at once ≈ 8–12 GB.** This is the real
  constraint, not money.
- **Mitigation:** Airflow and a live RAG query need not run simultaneously —
  start Astro, trigger the DAG, stop it; stop Ollama when not querying.
- **Disk:** Airflow image ≈ 1.5 GB + the project deps layer.
- **Cloud:** none. (Sub-project B will add BigQuery/dbt, which stay within free
  tiers — ~$0 with a $5 budget alert as backstop — documented in that spec.)

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Local Airflow runtime | Astronomer CLI (`astro dev`) | Industry-standard local Airflow; clean layout; resume signal |
| DAG execution model | `PythonOperator`/`@task` calling `ingest_directory` directly | Reuses tested code; cleanest code→orchestration wiring |
| DAG shape / schedule | 3 tasks (validate→ingest→verify), `@daily` + manual | Atomic-task + data-validation pattern; idempotent re-runs are safe |
| Service networking | `host.docker.internal`, metadata DB remapped to host port 5433 | Simplest on Mac; resolves the 5432 clash |
| Astro project location | `airflow/` subdirectory | Contains Astro's files; keeps repo root clean |
| Config injection | Env vars (`POSTGRES_HOST`, `OLLAMA_BASE_URL`) in Astro `.env` | Existing code already reads them; zero code change |
| GCP (sub-project B) | Personal project + BigQuery public dataset | Shareable, free-tier, keeps `aha-move` untouched |
