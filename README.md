# llm-vectordb-practice

Vector DB use cases with **pgvector + Ollama**, fully local.

## Use Cases

| Command | What it demonstrates |
|---|---|
| `python demo.py ingest [dir]` | Ingest a folder of .md/.txt/.pdf into chunked, embedded storage (default `data/corpus`) |
| `python demo.py compare "<query>"` | Compare vector-only vs hybrid (RRF) retrieval rankings side by side |
| `python demo.py semantic` | Semantic search over documents |
| `python demo.py rag` | RAG chatbot (retrieve + generate) |
| `python demo.py products` | Product / item similarity |
| `python demo.py logs` | Log anomaly detection via clustering |
| `python demo.py` | All four demos (semantic, rag, products, logs) |

## Stack

- **Vector DB**: PostgreSQL 16 + pgvector (HNSW index, cosine distance)
- **Embeddings**: Ollama `nomic-embed-text` (768-dim, local)
- **LLM**: Ollama `llama3.2` (local, used in RAG chatbot)
- **Ingestion**: recursive token-aware chunking (tiktoken `cl100k_base`, 600-token chunks, 80-token overlap), idempotent re-ingest via SHA-256 content hashes
- **Hybrid search**: pgvector cosine + Postgres full-text (`tsvector`), fused with Reciprocal Rank Fusion (k=60)

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
use_cases/
  semantic_search.py    # Index + query documents by meaning
  rag_chatbot.py        # Retrieve context + generate answer with LLM
  product_similarity.py # Find similar products by description
  log_clustering.py     # Detect anomalous log entries
data/
  corpus/        # sample corpus ingested by the RAG chatbot
  documents.py   # Sample document corpus
  products.py    # Sample product catalog
  logs.py        # Sample log stream with injected anomalies
tests/           # pytest suite (unit + `-m integration`)
demo.py          # CLI runner
```

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
