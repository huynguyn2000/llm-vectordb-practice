# Vector-DB Comparison (pgvector vs ChromaDB) — Design

**Date:** 2026-08-10
**Status:** Approved (autonomous execution under session goal)
**Context:** Abstracts vector retrieval behind a small `VectorBackend` protocol
and adds a **ChromaDB** backend alongside the existing **pgvector** one, so the
same corpus can be searched by either store and compared side by side. This is
the "swap the vector store" learning exercise; the abstraction is the reusable
value, and Weaviate is documented as the next drop-in adapter (not built — it
needs a Docker service; one comparison, not three DBs).

## Goals

- A `VectorBackend` protocol: `name`, `upsert(items)`, `search(embedding, top_k)
  -> list[dict]` (keys `id`, `content`, `source_path`, `chunk_index`, `score`).
- `PgvectorBackend` wrapping the existing `VectorStore.search_chunks`.
- `ChromaBackend` on **embedded ChromaDB** (local persistent or in-memory) —
  no Docker; uses our own `nomic-embed-text` embeddings for apples-to-apples.
- Populate Chroma from `data/corpus` (reuse loaders + chunker + Embedder).
- A `demo.py vsdb "<query>"` that searches both backends and prints top-k side
  by side (same shape as the hybrid `compare` demo).
- Additive: no change to the main RAG path (pgvector hybrid). Chroma is opt-in
  via a `chroma` extra.

## Non-goals

- Weaviate (documented as the next adapter; needs a Docker service).
- Porting hybrid (keyword) search to Chroma — the comparison is vector-only,
  which is the fair apples-to-apples axis (Chroma has no Postgres FTS).
- Replacing pgvector as the default store.

## Architecture & layout

```
vectorstores/
  __init__.py
  base.py        # VectorBackend Protocol + a shared result dataclass/dict shape
  pgvector.py    # PgvectorBackend(store): search wraps VectorStore.search_chunks
  chroma.py      # ChromaBackend: embedded chromadb; upsert + search
  corpus.py      # build_chroma_backend_from_corpus(dir, embedder) -> ChromaBackend
  tests/
    __init__.py
    test_chroma.py     # no-infra: in-memory chroma + deterministic vectors
    test_pgvector.py   # integration: pgvector backend search shape (real Postgres)
demo.py          # + `vsdb "<query>"` subcommand
pyproject.toml   # + `chroma` extra: chromadb>=0.5
README.md        # + Vector-DB comparison section (+ Weaviate-as-next-adapter note)
```

## VectorBackend protocol (`base.py`)

```python
class VectorBackend(Protocol):
    name: str
    def upsert(self, items: list[dict]) -> None: ...   # items: id/content/embedding/source_path/chunk_index
    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]: ...
```

`search` returns dicts with keys `id`, `content`, `source_path`, `chunk_index`,
`score` (cosine similarity, higher = better) — identical to
`VectorStore.search_chunks`, so both backends are interchangeable downstream.

## Backends

- **`PgvectorBackend(store: VectorStore)`** — `search` delegates to
  `store.search_chunks(embedding, top_k)` (already returns the right shape).
  `upsert` is a no-op / not needed (pgvector is populated by the ingestion
  pipeline); documented as such. `name = "pgvector"`.
- **`ChromaBackend`** — wraps a `chromadb` collection (embedded
  `PersistentClient(path=".chroma")` for the demo; `EphemeralClient` for tests).
  `upsert` adds `ids`, `documents=content`, `embeddings`, `metadatas={source_path,
  chunk_index}`. `search` calls `collection.query(query_embeddings=[embedding],
  n_results=top_k)` and maps results → the shared dict, converting Chroma's
  cosine **distance** to **similarity** (`score = 1 - distance`). Collection
  created with `metadata={"hnsw:space": "cosine"}`. `name = "chroma"`.

## Corpus population (`corpus.py`)

`build_chroma_backend_from_corpus(corpus_dir, embedder, client=None) -> ChromaBackend`:
walks the corpus, loads + chunks each file (reuse `ingestion.loaders`,
`ingestion.chunker`), embeds via `embedder.embed_many`, and upserts into a fresh
Chroma collection. Mirrors what the pgvector ingestion already did, so both
stores hold the same chunks.

## Demo (`demo.py vsdb "<query>"`)

Embeds the query once; searches `PgvectorBackend` (existing pgvector data) and a
Chroma backend (built from the corpus on first run / persisted), prints a
side-by-side top-k table (`rank | pgvector source#chunk | chroma source#chunk`).
Requires Postgres + Ollama + the `chroma` extra.

## Error handling

- `chroma` extra absent → `demo.py vsdb` prints a clear "install `.[chroma]`"
  message; `ChromaBackend` import is lazy so the rest of the package works
  without chromadb.
- Empty corpus / pgvector not ingested → clear message, non-zero exit.

## Testing

- `test_chroma.py` (**no infra**): use `chromadb`'s in-memory `EphemeralClient`
  and deterministic vectors (`tests.helpers.fake_embedding`). Upsert 3 chunks,
  search with a vector closest to one → assert it ranks first and the result
  dict has the 5 keys with `score` in [0,1]. Fast, deterministic, no
  Docker/Ollama. (Requires `chromadb` installed — skip with a clear reason if
  absent, mirroring the docling pattern.)
- `test_pgvector.py` (**integration**): `PgvectorBackend(store).search(...)`
  returns the shared shape (real Postgres, `zz-test-` fixture, purge).
- Existing suite untouched.

## Cost & resources

- **$0.** ChromaDB is embedded/local (no Docker); the `chroma` extra pulls
  chromadb (+ onnxruntime); modest. Base install unaffected.
- Note the possible dependency footprint (chromadb deps); keep it an opt-in
  extra so the default env stays lean (lesson from the Docling/dagster clash).

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Alternative store | ChromaDB (embedded) | Local, no Docker, easiest "swap" comparison |
| Weaviate | Documented next adapter, not built | Needs a Docker service; one comparison suffices |
| Comparison axis | Vector-only | Fair apples-to-apples; Chroma has no Postgres FTS/hybrid |
| Abstraction | `VectorBackend` protocol, shared dict shape | Backends interchangeable; reusable pattern |
| Delivery | `chroma` opt-in extra, lazy import | Default env stays lean (Docling/dagster lesson) |
| Chroma tests | In-memory EphemeralClient + fake vectors | Real, deterministic, no infra |
