# llm-vectordb-practice

Vector DB use cases with **pgvector + Ollama**, fully local.

## Use Cases

| Command | What it demonstrates |
|---|---|
| `python demo.py ingest [dir]` | Ingest a folder of .md/.txt/.pdf into chunked, embedded storage (default `data/corpus`) |
| `python demo.py semantic` | Semantic search over documents |
| `python demo.py rag` | RAG chatbot (retrieve + generate) |
| `python demo.py products` | Product / item similarity |
| `python demo.py logs` | Log anomaly detection via clustering |
| `python demo.py` | All four use cases |

## Stack

- **Vector DB**: PostgreSQL 16 + pgvector (HNSW index, cosine distance)
- **Embeddings**: Ollama `nomic-embed-text` (768-dim, local)
- **LLM**: Ollama `llama3.2` (local, used in RAG chatbot)
- **Ingestion**: recursive token-aware chunking (tiktoken `cl100k_base`, 600-token chunks, 80-token overlap), idempotent re-ingest via SHA-256 content hashes

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

**RAG Chatbot**: ingests `data/corpus` (chunked + embedded, skipping unchanged files by content hash), retrieves the most relevant chunks, then passes them with source citations as context to Ollama — the LLM only answers from retrieved context, not its own knowledge.

**Product Similarity**: embeds `name + description` together and searches by natural-language query — no keyword matching required.

**Log Anomaly Detection**: two modes — (1) similarity search to find logs like a reference, (2) outlier scoring where anomaly = high mean cosine distance from all other logs.
