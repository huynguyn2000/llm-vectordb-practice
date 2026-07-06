# Chunking + Real Ingestion — Design

**Date:** 2026-07-06
**Status:** Approved
**Context:** First milestone of the Month 1–2 goal ("Build one end-to-end RAG pipeline").
Replaces the hardcoded `data/documents.py` sample corpus with a real, idempotent
file-ingestion pipeline feeding chunked, embedded content into pgvector.

## Goals

- Ingest local files (`.md`, `.txt`, `.pdf`) from a directory into pgvector.
- Chunk documents with a recursive, token-aware splitter (the production default).
- Idempotent re-ingestion: unchanged files are skipped, changed files re-embedded,
  removed files cleaned up.
- Raw Python — no LangChain/LlamaIndex. A framework comparison variant is a
  separate, later milestone (planned alongside Ragas + observability).

## Non-goals

- Hybrid search (next milestone).
- Web/URL ingestion, OCR, structured (CSV/JSONL) sources.
- Airflow orchestration (later milestone; `ingest_directory` is designed to be
  callable from a DAG task).

## Architecture

```
ingestion/
  __init__.py
  loaders.py    # load_file(path) -> RawDocument (dispatch by extension)
  chunker.py    # chunk_text(text, chunk_size=600, overlap=80) -> list[Chunk]
  ingest.py     # ingest_directory(dir, store, embedder) -> IngestStats
core/
  models.py     # + RawDocument, Chunk, ChunkResult, IngestStats
  db.py         # + sources/chunks methods
init.sql        # + sources, chunks tables
demo.py         # + `python demo.py ingest <dir>` subcommand
data/corpus/    # small sample corpus: a few .md, one .txt, one small .pdf
tests/          # pytest suite (new)
```

**Data flow:** `ingest_directory` walks the folder → SHA-256 of raw bytes per file
→ compare against `sources` table → unchanged: skip; new/changed:
`load_file` → `chunk_text` → `embedder.embed_many` → upsert chunks in one
transaction per file; files in DB but missing on disk: delete source row
(CASCADE removes chunks).

**New dependencies:** `tiktoken` (token counting), `pypdf` (PDF text extraction).

## Data model

```sql
CREATE TABLE sources (
  id SERIAL PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,           -- relative path within corpus dir
  content_hash TEXT NOT NULL,          -- sha256 of file bytes
  ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  source_id INT REFERENCES sources(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,            -- position within the source doc
  content TEXT NOT NULL,
  token_count INT NOT NULL,
  embedding vector(768),
  UNIQUE (source_id, chunk_index)
);
-- HNSW index on chunks.embedding with vector_cosine_ops (same as documents)
```

The existing `documents` table stays; the semantic-search demo keeps working.
`use_cases/rag_chatbot.py` retrieval switches from `documents` to `chunks`
(results include `source path` for citation).

**Re-ingest logic per file:**
- hash match → skip
- changed → delete that source's chunks, re-chunk, re-embed, insert, update
  hash — all in a single transaction (a crash mid-file never leaves a
  half-ingested document)
- removed from disk → delete source row; CASCADE cleans chunks

## Chunker

Recursive split with separator hierarchy `["\n\n", "\n", ". ", " "]`:

- Pack whole paragraphs into chunks of ≤ 600 tokens (tiktoken `cl100k_base`).
- If a single unit exceeds the limit, descend to the next separator level.
- ~80-token overlap carried from the tail of each chunk into the next.
- Pure function, no I/O: `chunk_text(text, chunk_size=600, overlap=80)` →
  `list[Chunk]` with `content`, `token_count`, `chunk_index`.

This is the most heavily unit-tested component — it is the core learning
artifact of the milestone.

## Error handling

- Unreadable/corrupt file (e.g. encrypted PDF): warn with path, skip, count in
  `IngestStats.failed`. Never abort the run for a data problem.
- Empty/whitespace-only file: skip, count.
- Ollama or Postgres errors: fail fast with a clear message (infra problem,
  not data problem).
- End-of-run report: `IngestStats` — `ingested / skipped (unchanged) /
  deleted / failed`.

## Testing

pytest suite in `tests/`:

- **Chunker (unit, no infra):** boundary respect (no mid-paragraph splits when
  avoidable), overlap correctness, oversized-paragraph descent through
  separator levels, token limits honored, empty-input behavior.
- **Loaders (unit):** one test per file type; corrupt-PDF → raises cleanly.
- **Ingestion (integration, requires Postgres + Ollama):** run ingest twice →
  second run skips all files; modify one file → only it re-ingests; delete a
  file → its chunks disappear.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Source | Local files (.md/.txt/.pdf) | Simplest realistic source; extensible |
| Chunking | Recursive w/ overlap | Production default; teaches size/overlap/boundary tradeoffs |
| Size unit | Tokens (tiktoken) | Matches real embedding/LLM context limits |
| Re-ingest | Idempotent upsert by content hash | Production-realistic; cheap to demo |
| Framework | Raw code now, framework comparison later | Learn mechanism first; compare LlamaIndex variant in eval milestone |
| Structure | Dedicated `ingestion/` package | Clean seams for hybrid search + Airflow milestones |
