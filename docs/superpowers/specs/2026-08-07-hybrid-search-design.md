# Hybrid Search (vector + keyword) — Design

**Date:** 2026-08-07
**Status:** Approved
**Context:** Second milestone of the Month 1–2 RAG pipeline goal. Adds keyword
(lexical) retrieval alongside the existing pgvector semantic search and fuses
the two rankings with Reciprocal Rank Fusion, so the RAG chatbot retrieves on
both meaning and exact terms. Builds on the chunked-ingestion milestone
(`chunks` table, `search_chunks` vector retrieval).

## Goals

- Keyword retrieval over `chunks` using Postgres full-text search (`tsvector`).
- Fuse vector and keyword rankings with Reciprocal Rank Fusion (RRF).
- Keep vector-only retrieval available; add a demo that compares vector-only
  vs hybrid side by side.
- Raw Python — no LangChain/LlamaIndex, no new infrastructure (Postgres FTS
  only, no BM25 extension).

## Non-goals

- True BM25 ranking (needs the `pg_search`/ParadeDB extension) — out of scope;
  `ts_rank` is sufficient because RRF consumes rank position, not raw score.
- Cross-encoder re-ranking, query expansion, multilingual analyzers.
- Any change to the ingestion pipeline or the grounding prompt.

## Architecture

```
search/
  __init__.py
  fusion.py     # reciprocal_rank_fusion(ranked_id_lists, k=60) -> list[(id, score)]  PURE, no I/O
  hybrid.py     # hybrid_search(query, store, embedder, top_k=3, candidate_k=20) -> list[ChunkResult]
core/
  db.py         # + search_chunks_keyword(query_text, top_k); search_chunks (vector) unchanged
use_cases/
  rag_chatbot.py  # retrieve() -> hybrid_search; retrieve_vector() kept for the compare demo
demo.py         # + `compare "<query>"` subcommand
init.sql        # + content_tsv generated column + GIN index on chunks
tests/          # fusion unit tests + keyword/hybrid integration tests
```

**Query data flow:** `hybrid_search` embeds the query once → runs two
independent SQL queries against `chunks` (vector by cosine distance, keyword by
`tsvector` match), each returning up to `candidate_k` ranked rows →
`reciprocal_rank_fusion` merges the two ranked ID lists into one ranking → take
`top_k` → assemble `ChunkResult`s from rows already fetched (no third query).

**Isolation:** `fusion.py` knows nothing about Postgres or embeddings — it
takes ranked lists of IDs and returns a fused ranking. It is the most
heavily unit-tested unit, testable with zero infrastructure (like the chunker).

## Schema

A generated column keeps the lexical vector in sync with `content`
automatically — no trigger, no application code (requires PG12+; project runs
PG16).

```sql
ALTER TABLE chunks
  ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING GIN (content_tsv);
```

Both statements go into `init.sql` (guarded for fresh DBs). A one-time `ALTER`
is applied to the running dev DB; because the column is `GENERATED ... STORED`,
existing chunks are backfilled automatically — no re-ingestion required.

## Keyword search (`core/db.py`)

```sql
SELECT c.id, c.content, c.chunk_index, s.path AS source_path,
       ts_rank(c.content_tsv, websearch_to_tsquery('english', %s)) AS score
FROM chunks c JOIN sources s ON s.id = c.source_id
WHERE c.content_tsv @@ websearch_to_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s
```

- `websearch_to_tsquery` parses raw user input forgivingly (quotes, OR,
  `-`negation) and never raises on garbage — an unparseable/empty query yields
  a query that matches nothing.
- Returns the **same dict shape** as `search_chunks`
  (`id`, `content`, `chunk_index`, `source_path`, `score`) so fusion and
  `ChunkResult` assembly treat both engines identically.
- No lexical match → returns `[]`; RRF then degenerates to the vector ranking.

## RRF fusion (`search/fusion.py`)

```python
def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists. Each ID scores Σ 1/(k + rank) over the lists it
    appears in (rank is 0-based position). Returns (id, score) sorted by score
    descending, with deterministic tie-breaking by id. k=60 is the standard
    TREC default."""
```

Pure list math — no DB, no embeddings. Tie-breaking is deterministic (by id)
so tests and demo output are stable.

## Orchestration (`search/hybrid.py`)

```python
def hybrid_search(query, store, embedder, top_k=3, candidate_k=20) -> list[ChunkResult]:
    embedding = embedder.embed(query)
    vec_rows = store.search_chunks(embedding, top_k=candidate_k)
    kw_rows  = store.search_chunks_keyword(query, top_k=candidate_k)
    rows_by_id = {r["id"]: r for r in vec_rows + kw_rows}      # union, dedup by id
    fused = reciprocal_rank_fusion(
        [[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]]
    )
    return [ChunkResult(**{**rows_by_id[i], "score": s}) for i, s in fused[:top_k]]
```

`candidate_k=20` gives fusion a deep pool to reorder; `top_k=3` keeps the LLM
context tight. `ChunkResult.score` carries the RRF score for hybrid results.

## RAG integration (`use_cases/rag_chatbot.py`)

- `retrieve()` calls `hybrid_search`.
- The current vector-only body becomes `retrieve_vector()`, kept for the
  compare demo.
- The grounding prompt is unchanged, so
  `evals/rag_chatbot/promptfooconfig.yaml` still mirrors it.

## Compare demo (`demo.py compare "<query>"`)

Prints vector-only and hybrid rankings side by side so the improvement is
visible:

```
Query: "what is nomic-embed-text"
 rank  vector-only                     hybrid (RRF)
 1     photosynthesis.md#chunk0        vector-databases.pdf#chunk0
 2     ...                             ...
```

## Error handling & edge cases

- Empty/whitespace query → both engines return `[]` → `hybrid_search` returns
  `[]` → RAG refuses (existing behavior).
- Garbage keyword input → `websearch_to_tsquery` yields an empty query, no
  exception.
- A chunk returned by both engines is deduped by id (appears once, scored by
  its combined RRF contribution).
- Postgres/Ollama down → infra errors propagate (not swallowed), consistent
  with the ingestion milestone.

## Testing

- **Fusion (pure unit, no infra)** — the bulk of coverage: an ID ranked high in
  both lists outranks one ranked high in only one; an ID in a single list still
  appears; an empty input list → output equals the other list's order;
  single-list order is preserved regardless of `k`; deterministic tie-breaking.
- **Keyword search (integration, Postgres)** — an exact term present in exactly
  one chunk returns that chunk; a nonsense token returns `[]`; result dict shape
  matches `search_chunks`.
- **Hybrid (integration, Postgres)** — union dedups a chunk both engines return;
  a query with a distinctive keyword ranks that chunk higher than vector-only
  places it. Integration tests use the deterministic `FakeEmbedder` (no Ollama).

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Keyword backend | Postgres `tsvector`/`websearch_to_tsquery` | No new infra; stays in the existing pipeline |
| Fusion method | Reciprocal Rank Fusion (k=60) | Scale-free; no score normalization; industry default |
| Fusion location | Pure Python function over two SQL queries | Unit-testable with zero DB; algorithm is front-and-center |
| Vector-only path | Kept, plus a compare demo | Demonstrates *why* hybrid helps, not just claims it |
| tsvector storage | `GENERATED ... STORED` column + GIN | Always in sync, auto-backfills, no trigger/app code |
| Ranking function | `ts_rank` (not BM25) | RRF consumes rank position; BM25 needs an extra extension |
| candidate_k / top_k | 20 per engine / 3 into prompt | Deep pool for fusion to reorder, tight context for the LLM |
