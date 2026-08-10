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
- **Hybrid search**: pgvector cosine + Postgres full-text (`tsvector`), fused with Reciprocal Rank Fusion (k=60)
- **LangChain**: an LCEL RAG chain (`langchain-ollama`) over the same hybrid retrieval, with optional LangSmith tracing
- **Agent patterns**: LangGraph graphs — query routing and reflection (generate→critique→revise) — alongside the hybrid retrieval; ReAct (tool-calling) and supervisor/multi-agent are documented next patterns
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
  loaders.py     # file -> text (.md/.txt/.pdf)
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
agent_patterns/
  router.py      # LangGraph multi-route query classifier + answer generator
  reflection.py  # LangGraph reflection loop (generate → critique → revise)
  tests/         # Unit tests for router and reflection
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

## Agent Patterns

LangGraph builds agentic workflows that combine LLM reasoning with structured control flow.

**Router**: A multi-route classifier graph — takes a question, routes it to one of several LLM branches (e.g., database query, math reasoning, general knowledge), and returns a route label and answer. Run with `python demo.py router "<question>"`.

**Reflection**: A self-improving loop — generates a draft, critiques it, applies revisions if needed (up to a max count), and exits with an approval decision. Run with `python demo.py reflect "<task>"`.

Both graphs use the same `ChatOllama` instance and are testable with fake-LLM mocks. **ReAct** (Reasoning + Acting with tool calls) and **Supervisor** (multi-agent coordination) are natural next patterns — they require a tool-calling model, so they are documented but not built here.
