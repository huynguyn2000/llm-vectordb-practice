# DuckDB + dbt Structured Source — Design

**Date:** 2026-08-10
**Status:** Approved (autonomous execution under session goal — the milestone that completes it)
**Context:** The DE-layer sub-project B and the goal's finish line. Demonstrates
that **RAG sits on top of real data-engineering work**: a **DuckDB** warehouse
holds raw structured records, **dbt** (dbt-duckdb) transforms them into a clean
`documents` table, and those rows are exported as a text corpus the existing RAG
ingestion pipeline consumes. All local, `$0`, no Docker (DuckDB is in-process;
dbt-core runs locally).

## Goals

- A minimal dbt-duckdb project (`de/dbt/`): a seed of raw structured records →
  staging model → a `documents` mart (`id`, `source`, `content` text ready for RAG).
- `de/pipeline.py`: run dbt, read the `documents` table from DuckDB, and export
  each row to a text file corpus (`data/warehouse_corpus/`) — the DE→RAG bridge.
- `demo.py warehouse` runs the flow and prints a summary; the exported corpus
  can then be ingested by the existing `demo.py ingest data/warehouse_corpus`.
- Opt-in `de` extra (`dbt-duckdb`, `duckdb`); reuse the existing ingestion for
  the RAG side (no changes to core/search/ingestion/use_cases).

## Non-goals

- Dagster orchestration of dbt (documented as the integration point via
  `dagster-dbt`; not wired here — Dagster isn't on `main` and dbt+dagster in one
  venv risks a click/dep clash). 
- BigQuery/Redshift (replaced by DuckDB per the earlier cost/learning decision).
- Incremental models, snapshots, sources beyond the seed (YAGNI for the demo).

## Architecture & layout

```
de/
  __init__.py
  dbt/
    dbt_project.yml           # project "warehouse", duckdb profile
    profiles.yml              # duckdb target -> de/dbt/warehouse.duckdb
    seeds/faq.csv             # raw structured records (id, question, answer, topic)
    models/
      staging/stg_faq.sql     # clean/cast the seed
      marts/documents.sql     # -> id, source, content (question + answer as text)
  pipeline.py                 # run_dbt(); export_documents(duckdb_path) -> list[dict]; write_corpus(rows, out_dir)
  tests/
    __init__.py
    test_export.py            # no-dbt: build a tiny DuckDB, assert export/write logic
    test_dbt_build.py         # de extra: run dbt build, assert documents table rows/columns
demo.py                       # + `warehouse` subcommand
pyproject.toml                # + `de` extra: dbt-duckdb, duckdb
README.md                     # + Structured source (DuckDB + dbt) section
data/warehouse_corpus/        # generated (gitignored) — exported documents
```

## dbt project (`de/dbt/`)

- **Seed `faq.csv`**: ~8 rows of structured records — columns `id, question,
  answer, topic` (a small FAQ / knowledge base as the "raw warehouse data").
- **`stg_faq.sql`**: `select id, trim(question) as question, trim(answer) as
  answer, lower(topic) as topic from {{ ref('faq') }}`.
- **`documents.sql`** (mart): `select id, 'faq:' || id as source,
  question || '\n\n' || answer as content from {{ ref('stg_faq') }}` — the
  RAG-ready documents table (one text doc per record).
- `profiles.yml` uses the `duckdb` adapter with `path: warehouse.duckdb` (inside
  `de/dbt/`), schema `main`.

## Pipeline (`de/pipeline.py`)

- `run_dbt(project_dir="de/dbt") -> None`: subprocess `dbt build`
  (`--project-dir` / `--profiles-dir` = the dbt dir); raise a clear error if dbt
  isn't installed (name the `de` extra) or the build fails.
- `export_documents(duckdb_path) -> list[dict]`: `duckdb.connect(path)`, `SELECT
  id, source, content FROM documents ORDER BY id` → list of dicts.
- `write_corpus(rows, out_dir="data/warehouse_corpus") -> int`: write each row to
  `{out_dir}/{source_slug}.md` (content = the text); return count. This produces
  a corpus the existing `ingest_directory` can consume.

## Demo (`demo.py warehouse`)

Runs `run_dbt()` → `export_documents()` → `write_corpus()`, prints
"built N documents → data/warehouse_corpus/". Then the user can
`demo.py ingest data/warehouse_corpus` to push them through the real RAG
ingestion (chunk → embed → pgvector). Requires the `de` extra.

## Testing

- `test_export.py` (**no dbt, no infra**): create a tiny DuckDB in a tmp dir with
  the `duckdb` package (a hand-made `documents` table), assert `export_documents`
  returns the rows in the right shape and `write_corpus` writes the expected files
  with the content. Fast, deterministic. (Requires only `duckdb`; skip with a
  clear reason if the `de` extra is absent.)
- `test_dbt_build.py` (**de extra; no Postgres/Ollama**): run `run_dbt()` on the
  real seed in a tmp DuckDB, then `export_documents` and assert the `documents`
  table has one row per seed record with non-empty `content` containing the
  question text. DuckDB is in-process + dbt on a tiny seed is fast (seconds).
  Skip if `dbt`/`duckdb` absent.
- Existing suite untouched.

## Dependency-conflict watch

`dbt-core` pins `click`, `jinja2`, `agate`, etc.; there is some risk of a clash
with the broader stack (as Docling's antlr pin clashed with dagster). `de` is an
opt-in extra; after install, verify `pytest --collect-only` stays clean. If it
conflicts, document it (use a separate venv for the DE step) and keep the dev
venv green — same posture as prior milestones.

## Error handling

- `de` extra absent → `run_dbt`/`export_documents` raise a clear "install `.[de]`"
  error; `demo.py warehouse` catches and prints the hint.
- dbt build failure → surfaced with dbt's output.
- `data/warehouse_corpus/` is gitignored (generated artifact).

## Cost & resources

- **$0.** DuckDB is in-process; dbt-core + dbt-duckdb are light pip installs (no
  Docker, no torch). Fast. The heaviest external piece is only the optional RAG
  ingestion afterward (Ollama embeddings), which is the existing pipeline.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Warehouse | DuckDB (in-process) | Local, $0, no Docker — the chosen BigQuery/Redshift replacement |
| Transform | dbt-duckdb (seed → staging → documents mart) | Real dbt models; the DE craft on display |
| DE→RAG bridge | Export documents mart → text corpus → existing ingest | Shows RAG on top of DE work; reuses the tested pipeline |
| Dagster tie | Documented (`dagster-dbt`), not wired | Dagster not on main + dbt/dagster dep-clash risk; keep it feasible |
| Delivery | Opt-in `de` extra | Default env lean; dep-conflict containment |
| Tests | export (no-dbt) + dbt-build (de extra), both fast/local | DuckDB in-process → real verification without heavy infra |
