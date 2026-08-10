# llm-vectordb-practice

Vector DB use cases with **pgvector + Ollama**, fully local.

## Use Cases

| Command | What it demonstrates |
|---|---|
| `python demo.py ingest [dir]` | Ingest a folder of .md/.txt/.pdf into chunked, embedded storage (default `data/corpus`) |
| `python demo.py compare "<query>"` | Compare vector-only vs hybrid (RRF) retrieval rankings side by side |
| `python demo.py semantic` | Semantic search over documents |
| `python demo.py rag` | RAG chatbot (retrieve + generate) |
| `python demo.py langchain "<query>"` | RAG answer via the LangChain LCEL chain (same hybrid retrieval, local Ollama) |
| `python demo.py products` | Product / item similarity |
| `python demo.py logs` | Log anomaly detection via clustering |
| `python demo.py` | All four demos (semantic, rag, products, logs) |

## Stack

- **Vector DB**: PostgreSQL 16 + pgvector (HNSW index, cosine distance)
- **Embeddings**: Ollama `nomic-embed-text` (768-dim, local)
- **LLM**: Ollama `llama3.2` (local, used in RAG chatbot)
- **Ingestion**: recursive token-aware chunking (tiktoken `cl100k_base`, 600-token chunks, 80-token overlap), idempotent re-ingest via SHA-256 content hashes
- **Document parsing**: Docling (layout/table-aware → Markdown) for `.pdf`/`.docx` via the `docling` extra; falls back to `pypdf` for PDFs when the extra isn't installed
- **Hybrid search**: pgvector cosine + Postgres full-text (`tsvector`), fused with Reciprocal Rank Fusion (k=60)
- **LangChain**: an LCEL RAG chain (`langchain-ollama`) over the same hybrid retrieval, with optional LangSmith tracing
- **Orchestration**: Dagster — the ingestion pipeline as assets (`corpus_source → pgvector_chunks`) with an asset check and a daily schedule; run locally with `dagster dev`

## Quickstart

```bash
# 1. Copy env config
cp .env.example .env

# 2. Start Postgres + Ollama (pulls models on first run, takes a few minutes)
docker compose up -d

# 3. Install Python dependencies
pip install -e .

# 4. Run a use case
python demo.py semantic
python demo.py rag
python demo.py products
python demo.py logs
```

## Project Structure

```
core/
  db.py          # VectorStore — pgvector CRUD for all tables
  embedder.py    # Ollama embedding client
  models.py      # Pydantic models
ingestion/
  loaders.py     # file -> text (.md/.txt plain; .pdf/.docx via Docling, pypdf fallback)
  chunker.py     # text -> token-sized chunks with overlap
  ingest.py      # hash-diff orchestration: load -> chunk -> embed -> upsert
search/
  fusion.py      # reciprocal rank fusion (pure function)
  hybrid.py      # vector + keyword retrieval fused into one ranking
orchestration/
  resources.py   # Dagster resources: VectorStore + Embedder (dependency injection)
  assets.py      # corpus_source, pgvector_chunks assets + chunks_present check
  definitions.py # Dagster code location: assets, check, job, daily schedule
use_cases/
  semantic_search.py    # Index + query documents by meaning
  rag_chatbot.py        # Retrieve context + generate answer with LLM
  product_similarity.py # Find similar products by description
  log_clustering.py     # Detect anomalous log entries
langchain_rag/
  retriever.py   # BaseRetriever wrapping hybrid_search -> LangChain Documents
  chain.py       # LCEL RAG chain (retriever -> prompt -> ChatOllama)
data/
  corpus/        # sample corpus ingested by the RAG chatbot
  documents.py   # Sample document corpus
  products.py    # Sample product catalog
  logs.py        # Sample log stream with injected anomalies
tests/           # pytest suite (unit + `-m integration`)
demo.py          # CLI runner
```

## LangChain & LangSmith

A LangChain LCEL variant of the RAG pipeline reuses the same hybrid retrieval:

```bash
pip install -e ".[langchain]"
python demo.py langchain "what is machine learning"
```

To trace runs in **LangSmith**, set the env vars in `.env` (a free key from
smith.langchain.com) — tracing is off by default:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=llm-vectordb-practice
```

The chain is tagged (`run_name="langchain_rag"`, tags `rag`/`hybrid`) so traces
are legible in the LangSmith UI.

## Better parsing with Docling

By default, PDFs are parsed with `pypdf` (light, text-only). For layout- and
table-aware parsing (PDF/DOCX → Markdown), install the Docling extra:

```bash
pip install -e ".[docling]"     # pulls torch; first convert downloads ~GB of models
python demo.py ingest data/corpus
```

With the extra installed, `.pdf` and `.docx` are converted via Docling; without
it, `.pdf` still works via the pypdf fallback and `.docx` raises a clear error.

> **Heads-up — install Docling in a separate venv.** Docling pins
> `antlr4-python3-runtime` 4.9.x, which is incompatible with Dagster's generated
> asset-selection lexer (fails with `TypeError: ord() ...`). So the `docling`
> and `orchestration` extras can't share one environment. Use a dedicated venv
> for Docling-based ingestion; keep the default/dev venv (with Dagster, tests)
> Docling-free — the pypdf fallback keeps PDFs working there.

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

## Testing

```bash
pytest                    # unit tests (no infra needed)
pytest -m integration     # DB tests — needs `docker compose up -d` (no Ollama needed; tests use a fake embedder)
```

## Key Concepts

**Semantic Search**: embeds a query and retrieves the `top_k` documents by cosine similarity.

**RAG Chatbot**: ingests `data/corpus`, then retrieves with **hybrid search** — pgvector semantic search and Postgres keyword search fused by Reciprocal Rank Fusion — and passes the top chunks with source citations to Ollama. `python demo.py compare "<query>"` shows the vector-only vs hybrid rankings side by side.

**Product Similarity**: embeds `name + description` together and searches by natural-language query — no keyword matching required.

**Log Anomaly Detection**: two modes — (1) similarity search to find logs like a reference, (2) outlier scoring where anomaly = high mean cosine distance from all other logs.
