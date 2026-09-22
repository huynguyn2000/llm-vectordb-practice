# llm-vectordb-practice

Vector DB use cases with **pgvector + Ollama**, fully local.

## Use Cases

| Command | What it demonstrates |
|---|---|
| `python demo.py ingest [dir]` | Ingest a folder of .md/.txt/.pdf into chunked, embedded storage (default `data/corpus`) |
| `python demo.py compare "<query>"` | Compare vector-only vs hybrid (RRF) retrieval rankings side by side |
| `python demo.py vsdb "<query>"` | Compare pgvector vs ChromaDB vector search side by side |
| `python demo.py semantic` | Semantic search over documents |
| `python demo.py rag` | RAG chatbot (retrieve + generate) |
| `python demo.py langchain "<query>"` | RAG answer via the LangChain LCEL chain (same hybrid retrieval, local Ollama) |
| `python demo.py router "<q>"` | Query routing (classify question type, generate answer per route) |
| `python demo.py reflect "<task>"` | Reflection agent (generate draft → critique → revise, with approval gate) |
| `python demo.py products` | Product / item similarity |
| `python demo.py logs` | Log anomaly detection via clustering |
| `python demo.py` | All four demos (semantic, rag, products, logs) |

## Stack

- **Vector DB**: PostgreSQL 16 + pgvector (HNSW index, cosine distance)
- **Embeddings**: Ollama `nomic-embed-text` (768-dim, local)
- **LLM**: Ollama `llama3.2` (local, used in RAG chatbot)
- **Ingestion**: recursive token-aware chunking (tiktoken `cl100k_base`, 600-token chunks, 80-token overlap), idempotent re-ingest via SHA-256 content hashes
- **Vector-store abstraction**: a `VectorBackend` protocol with pgvector and ChromaDB (embedded) backends; `demo.py vsdb` compares them (Weaviate is a documented next adapter)
- **Document parsing**: Docling (layout/table-aware → Markdown) for `.pdf`/`.docx` via the `docling` extra; falls back to `pypdf` for PDFs when the extra isn't installed
- **Hybrid search**: pgvector cosine + Postgres full-text (`tsvector`), fused with Reciprocal Rank Fusion (k=60)
- **LangChain**: an LCEL RAG chain (`langchain-ollama`) over the same hybrid retrieval, with optional LangSmith tracing
- **Agent patterns**: LangGraph graphs — query routing and reflection (generate→critique→revise) — alongside the hybrid retrieval; ReAct (tool-calling) and supervisor/multi-agent are documented next patterns
- **Orchestration**: Dagster — the ingestion pipeline as assets (`corpus_source → pgvector_chunks`) with an asset check and a daily schedule; run locally with `dagster dev`
- **Evaluation**: Ragas — faithfulness / answer-relevancy / context-precision / context-recall over a labeled Q&A set, comparing vector-only vs hybrid (local Ollama judge by default)

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
vectorstores/
  base.py        # VectorBackend protocol (abstraction for vector stores)
  pgvector.py    # PgvectorBackend — queries pgvector in PostgreSQL
  chroma.py      # ChromaBackend — embedded ChromaDB
  corpus.py      # build_chroma_backend_from_corpus utility
orchestration/
  resources.py   # Dagster resources: VectorStore + Embedder (dependency injection)
  assets.py      # corpus_source, pgvector_chunks assets + chunks_present check
  definitions.py # Dagster code location: assets, check, job, daily schedule
use_cases/
  semantic_search.py    # Index + query documents by meaning
  rag_chatbot.py        # Retrieve context + generate answer with LLM
  product_similarity.py # Find similar products by description
  log_clustering.py     # Detect anomalous log entries
agent_patterns/
  router.py      # LangGraph multi-route query classifier + answer generator
  reflection.py  # LangGraph reflection loop (generate → critique → revise)
  tests/         # Unit tests for router and reflection
langchain_rag/
  retriever.py   # BaseRetriever wrapping hybrid_search -> LangChain Documents
  chain.py       # LCEL RAG chain (retriever -> prompt -> ChatOllama)
evals/ragas/
  dataset.json        # hand-labeled Q&A (question + ground truth)
  judge.py            # local-Ollama (default) / API judge factory for Ragas
  run_eval.py         # scores vector-only vs hybrid, writes results.json
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

## Vector-DB Comparison

This project includes a `VectorBackend` abstraction that allows comparing different vector databases side by side. Currently supported:

- **pgvector** (PostgreSQL extension) — production-ready, HNSW index, runs on existing Postgres infrastructure
- **ChromaDB** (embedded) — lightweight, file-based, good for prototyping and offline scenarios

Compare search rankings with:

```bash
pip install -e ".[chroma]"                      # Install chromadb extra
python demo.py vsdb "what are vector databases for"
```

The `demo.py vsdb` command embeds the query once and retrieves top-5 results from both backends, showing source file, chunk index, and similarity score. Future adapters (e.g. Weaviate) slot in as additional `VectorBackend` implementations.

## Testing

```bash
pytest                    # unit tests (no infra needed)
pytest -m integration     # DB tests — needs `docker compose up -d` (no Ollama needed; tests use a fake embedder)
```

## Evaluation (Ragas)

Measure retrieval + generation quality and compare retrieval modes:

```bash
docker compose up -d
pip install -e ".[eval]"
python demo.py ingest data/corpus
python -m evals.ragas.run_eval            # add --limit 2 for a quick run
```

It scores faithfulness, answer-relevancy, context-precision, and context-recall
for **vector-only vs hybrid** retrieval over `evals/ragas/dataset.json`, prints a
side-by-side table, and writes `results.json`. The judge defaults to local
`llama3.2` ($0) — set `RAGAS_JUDGE=openai` (or `anthropic`) with the provider's
API key for more stable scores. Local-judge scores are noisy — read them as
relative, not authoritative.

**Note on speed:** the local `llama3.2` judge is fine for wiring/plumbing but is impractically slow for a full run (≈2–3 min per metric call). For real numbers, use an API judge: set `RAGAS_JUDGE=anthropic` (or `openai`) with the provider's API key. Locally, use `--limit` for a small smoke run only.

## Key Concepts

**Semantic Search**: embeds a query and retrieves the `top_k` documents by cosine similarity.

**RAG Chatbot**: ingests `data/corpus`, then retrieves with **hybrid search** — pgvector semantic search and Postgres keyword search fused by Reciprocal Rank Fusion — and passes the top chunks with source citations to Ollama. `python demo.py compare "<query>"` shows the vector-only vs hybrid rankings side by side.

**Product Similarity**: embeds `name + description` together and searches by natural-language query — no keyword matching required.

**Log Anomaly Detection**: two modes — (1) similarity search to find logs like a reference, (2) outlier scoring where anomaly = high mean cosine distance from all other logs.

## Agent Patterns

LangGraph builds agentic workflows that combine LLM reasoning with structured control flow.

**Router**: A multi-route classifier graph — takes a question, routes it to one of several LLM branches (e.g., database query, math reasoning, general knowledge), and returns a route label and answer. Run with `python demo.py router "<question>"`.

**Reflection**: A self-improving loop — generates a draft, critiques it, applies revisions if needed (up to a max count), and exits with an approval decision. Run with `python demo.py reflect "<task>"`.

Both graphs use the same `ChatOllama` instance and are testable with fake-LLM mocks. **ReAct** (Reasoning + Acting with tool calls) and **Supervisor** (multi-agent coordination) are natural next patterns — they require a tool-calling model, so they are documented but not built here.
