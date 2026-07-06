# Chunking + Real Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded sample corpus with an idempotent file-ingestion pipeline (`.md`/`.txt`/`.pdf` → token-aware chunks → embeddings → pgvector) that the RAG chatbot retrieves from.

**Architecture:** New `ingestion/` package with three focused modules — `loaders.py` (file → text), `chunker.py` (text → token-sized chunks, pure function), `ingest.py` (hash-diff orchestration). Two new tables (`sources`, `chunks`) with per-file transactional upsert; unchanged files are skipped by SHA-256 content hash. `use_cases/rag_chatbot.py` retrieval switches from `documents` to `chunks`.

**Tech Stack:** Python 3.11+, psycopg2 + pgvector (raw SQL, no ORM), tiktoken (`cl100k_base`), pypdf, Ollama embeddings, pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-chunking-ingestion-design.md`

## Global Constraints

- No LangChain/LlamaIndex — raw Python only (spec decision).
- Chunk defaults: `chunk_size=600` tokens, `overlap=80` tokens, tokenizer `cl100k_base`.
- Chunks must never exceed `chunk_size` tokens.
- Separator hierarchy, exact: `["\n\n", "\n", ". ", " "]`.
- Embedding dimension is 768 (`nomic-embed-text`), matching existing tables.
- The existing `documents`, `products`, `logs` tables and their use cases must keep working untouched.
- `sources.path` stores the path **relative to the ingested corpus directory**.
- All DB work for one file happens in a single transaction (crash never leaves a half-ingested file).
- Data problems (corrupt/empty file) are warn-and-continue; infra problems (Postgres/Ollama down) fail fast by propagating.
- Run everything from the repo root with the venv active: `source .venv/bin/activate`.
- Integration tests are marked `@pytest.mark.integration` and require `docker compose up -d` (Postgres only — they use a fake embedder, no Ollama needed).
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Schema + dependencies

**Files:**
- Modify: `init.sql` (append new tables)
- Modify: `pyproject.toml` (deps, packages, pytest markers)

**Interfaces:**
- Produces: `sources` and `chunks` tables in Postgres; `tiktoken` and `pypdf` importable.

- [ ] **Step 1: Append the new tables to `init.sql`**

Add to the end of `init.sql`:

```sql
CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    source_id INT REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    embedding vector(768),
    UNIQUE (source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
```

- [ ] **Step 2: Update `pyproject.toml`**

In `[project] dependencies`, add:

```toml
    "tiktoken>=0.7",
    "pypdf>=4.0",
```

Change the packages line to:

```toml
[tool.setuptools]
packages = ["core", "use_cases", "data", "ingestion"]
```

Add at the end of the file:

```toml
[tool.pytest.ini_options]
markers = ["integration: requires a running Postgres (docker compose up -d)"]
```

- [ ] **Step 3: Create the (empty for now) `ingestion` package so pip install succeeds**

```bash
mkdir -p ingestion && touch ingestion/__init__.py
```

- [ ] **Step 4: Install and apply the schema**

```bash
pip install -e ".[dev]"
docker compose up -d postgres
docker compose exec -T postgres psql -U vectordb -d vectordb -f /docker-entrypoint-initdb.d/init.sql
```

Expected: `CREATE TABLE` / `CREATE INDEX` for the new objects, `NOTICE: relation "..." already exists, skipping` for the old ones. (init.sql only auto-runs on a fresh volume, hence the manual apply; everything is `IF NOT EXISTS` so re-running is safe. If `.env` overrides `POSTGRES_USER`/`POSTGRES_DB`, substitute those values.)

- [ ] **Step 5: Verify tables exist**

```bash
docker compose exec -T postgres psql -U vectordb -d vectordb -c "\d chunks"
```

Expected: table description showing `source_id`, `chunk_index`, `content`, `token_count`, `embedding | vector(768)` and the `chunks_embedding_idx` hnsw index.

- [ ] **Step 6: Commit**

```bash
git add init.sql pyproject.toml ingestion/__init__.py
git commit -m "feat: add sources/chunks schema and ingestion deps (tiktoken, pypdf)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Token-aware recursive chunker

**Files:**
- Modify: `core/models.py` (add `Chunk`)
- Create: `ingestion/chunker.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_chunker.py`

**Interfaces:**
- Produces: `chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[Chunk]` and `count_tokens(text: str) -> int` in `ingestion.chunker`; `Chunk(content: str, token_count: int, chunk_index: int)` in `core.models`. Task 5 calls `chunk_text` with defaults.

- [ ] **Step 1: Add the `Chunk` model to `core/models.py`**

Append:

```python
class Chunk(BaseModel):
    content: str
    token_count: int
    chunk_index: int
```

- [ ] **Step 2: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_chunker.py`:

```python
from ingestion.chunker import chunk_text, count_tokens


def test_count_tokens_positive():
    assert 0 < count_tokens("hello world") <= len("hello world")


def test_empty_and_whitespace_input_yield_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_unmodified_chunk():
    text = "Para one.\n\nPara two."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count == count_tokens(text)


def test_chunks_never_exceed_chunk_size():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_chunk_indexes_are_sequential():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_splits_fall_on_sentence_boundaries():
    # One giant paragraph (no \n\n) forces descent to the ". " level.
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(c.content.endswith(".") for c in chunks)


def test_consecutive_chunks_share_overlap():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        first_sentence_of_b = b.content.split(".")[0] + "."
        assert first_sentence_of_b in a.content


def test_zero_overlap_means_disjoint_chunks():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    for a, b in zip(chunks, chunks[1:]):
        first_sentence_of_b = b.content.split(".")[0] + "."
        assert first_sentence_of_b not in a.content


def test_unsplittable_text_hard_splits_by_tokens():
    text = "x" * 5000  # no separators at all
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.chunker'` (collection error).

- [ ] **Step 4: Implement the chunker**

Create `ingestion/chunker.py`:

```python
"""Recursive, token-aware text chunker.

Splits text into chunks of at most `chunk_size` tokens, preferring coarse
boundaries (paragraphs -> lines -> sentences -> words) and only descending
a level when a piece is still too large. Consecutive chunks share up to
`overlap` tokens of trailing context.
"""

import tiktoken

from core.models import Chunk

_ENC = tiktoken.get_encoding("cl100k_base")

SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_keeping_sep(text: str, sep: str) -> list[str]:
    """Split on sep, keeping it attached to the preceding part, so that
    concatenating the parts reproduces the original text."""
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _split_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces of at most chunk_size tokens each, using the
    coarsest separator that gets every piece under the limit."""
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        # Nothing left to split on: hard-split by tokens.
        tokens = _ENC.encode(text)
        return [
            _ENC.decode(tokens[i : i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]
    sep, rest = separators[0], separators[1:]
    pieces: list[str] = []
    for part in _split_keeping_sep(text, sep):
        if not part.strip():
            continue
        if count_tokens(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_pieces(part, chunk_size, rest))
    return pieces


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[Chunk]:
    """Chunk text into <= chunk_size-token chunks with ~overlap-token overlap.

    Overlap is piece-aligned: the trailing pieces of a finished chunk (up to
    `overlap` tokens' worth) seed the next chunk, so no chunk exceeds
    chunk_size tokens.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pieces = _split_pieces(text, chunk_size, SEPARATORS)

    raw_chunks: list[str] = []
    window: list[str] = []  # pieces of the chunk being built
    window_tokens = 0
    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if window and window_tokens + piece_tokens > chunk_size:
            raw_chunks.append("".join(window))
            # Keep at most `overlap` tokens of tail pieces as shared context.
            while window and window_tokens > overlap:
                window_tokens -= count_tokens(window[0])
                window.pop(0)
        window.append(piece)
        window_tokens += piece_tokens
    if window:
        raw_chunks.append("".join(window))

    return [
        Chunk(content=c.strip(), token_count=count_tokens(c.strip()), chunk_index=i)
        for i, c in enumerate(raw_chunks)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_chunker.py -v`
Expected: all 10 tests PASS. (First run downloads the `cl100k_base` encoding file — needs network once.)

- [ ] **Step 6: Commit**

```bash
git add core/models.py ingestion/chunker.py tests/__init__.py tests/test_chunker.py
git commit -m "feat: recursive token-aware chunker with piece-aligned overlap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: File loaders (.md / .txt / .pdf)

**Files:**
- Modify: `core/models.py` (add `RawDocument`)
- Create: `ingestion/loaders.py`
- Create: `tests/helpers.py` (minimal PDF builder)
- Test: `tests/test_loaders.py`

**Interfaces:**
- Produces: `load_file(path: Path) -> RawDocument`, `SUPPORTED_EXTENSIONS: set[str]` (`{".md", ".txt", ".pdf"}`), `class LoaderError(Exception)` in `ingestion.loaders`; `RawDocument(path: str, text: str)` in `core.models`; `build_pdf(text: str) -> bytes` in `tests.helpers`. Task 5 consumes all three loader names; Task 6 uses `build_pdf` to generate the corpus PDF.

- [ ] **Step 1: Add the `RawDocument` model to `core/models.py`**

Append:

```python
class RawDocument(BaseModel):
    path: str
    text: str
```

- [ ] **Step 2: Write the PDF test helper**

Create `tests/helpers.py`:

```python
"""Shared test utilities: a stdlib-only minimal PDF writer and a
deterministic fake embedder (no Ollama needed)."""


def build_pdf(text: str) -> bytes:
    """Build a tiny one-page PDF containing `text` (Helvetica, no escaping —
    keep text free of parentheses and backslashes)."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_pos,
    )
    return bytes(out)
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_loaders.py`:

```python
import pytest

from ingestion.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_file
from tests.helpers import build_pdf


def test_supported_extensions():
    assert SUPPORTED_EXTENSIONS == {".md", ".txt", ".pdf"}


def test_load_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody text.", encoding="utf-8")
    doc = load_file(p)
    assert doc.text == "# Title\n\nBody text."
    assert doc.path == str(p)


def test_load_txt(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Plain text.", encoding="utf-8")
    assert load_file(p).text == "Plain text."


def test_load_pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf("Hello from a PDF document."))
    assert "Hello from a PDF document." in load_file(p).text


def test_corrupt_pdf_raises_loader_error(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf")
    with pytest.raises(LoaderError):
        load_file(p)


def test_unsupported_extension_raises_loader_error(tmp_path):
    p = tmp_path / "doc.docx"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(LoaderError):
        load_file(p)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.loaders'`.

- [ ] **Step 5: Implement the loaders**

Create `ingestion/loaders.py`:

```python
"""File -> text loaders. One function per format, dispatched by extension."""

from pathlib import Path

from pypdf import PdfReader

from core.models import RawDocument

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


class LoaderError(Exception):
    """A file could not be loaded (unsupported type or unreadable content)."""


def load_file(path: Path) -> RawDocument:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        try:
            reader = PdfReader(path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise LoaderError(f"cannot read PDF {path}: {exc}") from exc
    else:
        raise LoaderError(f"unsupported file type: {path}")
    return RawDocument(path=str(path), text=text)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_loaders.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add core/models.py ingestion/loaders.py tests/helpers.py tests/test_loaders.py
git commit -m "feat: file loaders for md/txt/pdf with stdlib PDF test fixture

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: VectorStore chunk methods

**Files:**
- Modify: `core/db.py` (append methods to `VectorStore`)
- Modify: `tests/helpers.py` (add fake embedder)
- Create: `tests/conftest.py`
- Test: `tests/test_db_chunks.py`

**Interfaces:**
- Produces on `VectorStore`: `get_source_hashes() -> dict[str, str]`, `upsert_source_with_chunks(path: str, content_hash: str, chunks: list[tuple[str, int, list[float]]]) -> None` (tuples are `(content, token_count, embedding)`), `delete_source(path: str) -> None`, `search_chunks(embedding: list[float], top_k: int = 5) -> list[dict]` (dict keys: `id`, `content`, `chunk_index`, `source_path`, `score`). Also `fake_embedding(text: str, dim: int = 768) -> list[float]` and `class FakeEmbedder` (with `embed`/`embed_many` matching `core.embedder.Embedder`) in `tests.helpers`; a `store` fixture in `tests/conftest.py` that skips when Postgres is down. Tasks 5–6 consume all of these.

- [ ] **Step 1: Add the fake embedder to `tests/helpers.py`**

Append:

```python
import hashlib
import struct


def fake_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic pseudo-embedding: same text -> same vector. Lets DB and
    ingestion tests run without Ollama."""
    digest = hashlib.sha256(text.encode()).digest()
    data = digest * (dim * 4 // len(digest) + 1)
    return [
        struct.unpack_from("<I", data, i * 4)[0] % 1000 / 1000.0 for i in range(dim)
    ]


class FakeEmbedder:
    """Drop-in for core.embedder.Embedder in tests."""

    def embed(self, text: str) -> list[float]:
        return fake_embedding(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import pytest

from core.db import VectorStore


@pytest.fixture
def store():
    try:
        s = VectorStore()
    except Exception:
        pytest.skip("Postgres not reachable - start it with `docker compose up -d`")
    yield s
    s.close()
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_db_chunks.py`. Test rows use the `zz-test-` path prefix so they can never collide with real corpus paths; every test cleans up after itself.

```python
import pytest

from tests.helpers import fake_embedding

pytestmark = pytest.mark.integration

PATH_A = "zz-test-db/a.md"


@pytest.fixture(autouse=True)
def cleanup(store):
    store.delete_source(PATH_A)
    yield
    store.delete_source(PATH_A)


def _rows(texts):
    return [(t, 3, fake_embedding(t)) for t in texts]


def test_upsert_and_hash_roundtrip(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha", "beta"]))
    assert store.get_source_hashes()[PATH_A] == "hash1"


def test_upsert_replaces_previous_chunks(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha", "beta"]))
    store.upsert_source_with_chunks(PATH_A, "hash2", _rows(["gamma"]))
    assert store.get_source_hashes()[PATH_A] == "hash2"
    ours = [
        r
        for r in store.search_chunks(fake_embedding("gamma"), top_k=50)
        if r["source_path"] == PATH_A
    ]
    assert [r["content"] for r in ours] == ["gamma"]


def test_delete_source_cascades_to_chunks(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha"]))
    store.delete_source(PATH_A)
    assert PATH_A not in store.get_source_hashes()
    results = store.search_chunks(fake_embedding("alpha"), top_k=50)
    assert all(r["source_path"] != PATH_A for r in results)


def test_search_chunks_result_shape(store):
    store.upsert_source_with_chunks(PATH_A, "hash1", _rows(["alpha"]))
    ours = [
        r
        for r in store.search_chunks(fake_embedding("alpha"), top_k=50)
        if r["source_path"] == PATH_A
    ]
    assert set(ours[0]) == {"id", "content", "chunk_index", "source_path", "score"}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_db_chunks.py -v`
Expected: FAIL — `AttributeError: 'VectorStore' object has no attribute 'delete_source'` (or skip if Postgres is down — start it first).

- [ ] **Step 5: Implement the methods**

Append to the `VectorStore` class in `core/db.py`:

```python
    # --- Sources & chunks (file ingestion pipeline) ---

    def get_source_hashes(self) -> dict[str, str]:
        """path -> content_hash for every ingested source."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT path, content_hash FROM sources")
            return {path: content_hash for path, content_hash in cur.fetchall()}

    def upsert_source_with_chunks(
        self,
        path: str,
        content_hash: str,
        chunks: list[tuple[str, int, list[float]]],
    ) -> None:
        """Replace a source's chunks atomically: upsert the source row,
        delete its old chunks, insert the new (content, token_count,
        embedding) tuples - all in one transaction."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (path, content_hash)
                VALUES (%s, %s)
                ON CONFLICT (path)
                DO UPDATE SET content_hash = EXCLUDED.content_hash, ingested_at = NOW()
                RETURNING id
                """,
                (path, content_hash),
            )
            source_id = cur.fetchone()[0]
            cur.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
            for idx, (content, token_count, embedding) in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO chunks (source_id, chunk_index, content, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (source_id, idx, content, token_count, embedding),
                )
        self.conn.commit()

    def delete_source(self, path: str) -> None:
        """Remove a source; its chunks go with it via ON DELETE CASCADE."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM sources WHERE path = %s", (path,))
        self.conn.commit()

    def search_chunks(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id, c.content, c.chunk_index, s.path AS source_path,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c
                JOIN sources s ON s.id = c.source_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            return [dict(r) for r in cur.fetchall()]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_db_chunks.py -v`
Expected: 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add core/db.py tests/helpers.py tests/conftest.py tests/test_db_chunks.py
git commit -m "feat: VectorStore source/chunk methods with transactional upsert

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Ingestion orchestrator + demo subcommand

**Files:**
- Modify: `core/models.py` (add `IngestStats`)
- Create: `ingestion/ingest.py`
- Modify: `demo.py` (add `ingest` subcommand)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunk_text` (Task 2), `load_file`/`SUPPORTED_EXTENSIONS`/`LoaderError` (Task 3), `VectorStore` chunk methods + `FakeEmbedder`/`build_pdf`/`store` fixture (Tasks 3–4).
- Produces: `ingest_directory(directory: str | Path, store: VectorStore, embedder: Embedder) -> IngestStats` in `ingestion.ingest`; `IngestStats(ingested: int, skipped: int, deleted: int, failed: int)` in `core.models`; `python demo.py ingest [dir]` CLI. Task 6 calls `ingest_directory("data/corpus", store, embedder)`.

- [ ] **Step 1: Add the `IngestStats` model to `core/models.py`**

Append:

```python
class IngestStats(BaseModel):
    ingested: int = 0
    skipped: int = 0
    deleted: int = 0
    failed: int = 0
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ingest.py`:

```python
import pytest

from ingestion.ingest import ingest_directory
from tests.helpers import FakeEmbedder, build_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "zz-test-notes.md").write_text(
        "# Notes\n\nSome markdown notes about testing.", encoding="utf-8"
    )
    (tmp_path / "zz-test-plain.txt").write_text(
        "Plain text content for the ingest test.", encoding="utf-8"
    )
    (tmp_path / "zz-test-doc.pdf").write_bytes(build_pdf("Hello from a test PDF."))
    (tmp_path / "ignored.docx").write_text("should be ignored", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def clean_store(store):
    def _purge():
        for path in list(store.get_source_hashes()):
            if path.startswith("zz-test"):
                store.delete_source(path)

    _purge()
    yield store
    _purge()


def test_first_ingest_ingests_all_supported_files(store, corpus):
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.ingested == 3
    assert stats.skipped == stats.deleted == stats.failed == 0


def test_second_ingest_skips_everything(store, corpus):
    ingest_directory(corpus, store, FakeEmbedder())
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.skipped == 3
    assert stats.ingested == stats.deleted == stats.failed == 0


def test_modified_file_is_reingested(store, corpus):
    ingest_directory(corpus, store, FakeEmbedder())
    (corpus / "zz-test-notes.md").write_text(
        "# Notes\n\nEdited content.", encoding="utf-8"
    )
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.ingested == 1
    assert stats.skipped == 2


def test_removed_file_is_cleaned_up(store, corpus):
    ingest_directory(corpus, store, FakeEmbedder())
    (corpus / "zz-test-plain.txt").unlink()
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.deleted == 1
    assert stats.skipped == 2
    assert "zz-test-plain.txt" not in store.get_source_hashes()


def test_corrupt_pdf_is_counted_failed_and_run_continues(store, corpus):
    (corpus / "zz-test-broken.pdf").write_bytes(b"not a real pdf")
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.failed == 1
    assert stats.ingested == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.ingest'`.

- [ ] **Step 4: Implement the orchestrator**

Create `ingestion/ingest.py`:

```python
"""Idempotent directory ingestion: hash-diff against the sources table,
then load -> chunk -> embed -> upsert per changed file."""

import hashlib
from pathlib import Path

from core.db import VectorStore
from core.embedder import Embedder
from core.models import IngestStats
from ingestion.chunker import chunk_text
from ingestion.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_file


def ingest_directory(
    directory: str | Path, store: VectorStore, embedder: Embedder
) -> IngestStats:
    root = Path(directory)
    stats = IngestStats()
    known = store.get_source_hashes()
    seen: set[str] = set()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rel = str(path.relative_to(root))
        seen.add(rel)

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if known.get(rel) == content_hash:
            stats.skipped += 1
            continue

        try:
            doc = load_file(path)
        except LoaderError as exc:
            print(f"WARNING: skipping {rel}: {exc}")
            stats.failed += 1
            continue

        chunks = chunk_text(doc.text)
        if not chunks:
            print(f"WARNING: skipping {rel}: no text content")
            stats.failed += 1
            continue

        embeddings = embedder.embed_many([c.content for c in chunks])
        store.upsert_source_with_chunks(
            rel,
            content_hash,
            [(c.content, c.token_count, e) for c, e in zip(chunks, embeddings)],
        )
        stats.ingested += 1

    # Sources that vanished from disk since the last run.
    for stale in set(known) - seen:
        store.delete_source(stale)
        stats.deleted += 1

    return stats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 5 tests PASS.

- [ ] **Step 6: Add the `ingest` subcommand to `demo.py`**

In `demo.py`, insert at the top of `main()` (before the `if selected not in USE_CASES` check):

```python
    if selected == "ingest":
        from ingestion.ingest import ingest_directory

        corpus_dir = sys.argv[2] if len(sys.argv) > 2 else "data/corpus"
        with VectorStore() as store:
            stats = ingest_directory(corpus_dir, store, Embedder())
        print(
            f"Ingested: {stats.ingested}  skipped (unchanged): {stats.skipped}  "
            f"deleted: {stats.deleted}  failed: {stats.failed}"
        )
        return
```

And extend the module docstring usage block with:

```
  python demo.py ingest [dir]       # ingest a folder of .md/.txt/.pdf (default: data/corpus)
```

- [ ] **Step 7: Run the whole suite**

Run: `pytest -v`
Expected: all chunker/loader/db/ingest tests PASS.

- [ ] **Step 8: Commit**

```bash
git add core/models.py ingestion/ingest.py demo.py tests/test_ingest.py
git commit -m "feat: idempotent ingest_directory with hash-diff and demo subcommand

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Sample corpus + RAG chatbot on chunks

**Files:**
- Modify: `core/models.py` (add `ChunkResult`)
- Create: `data/corpus/machine-learning.md`, `data/corpus/photosynthesis.md`, `data/corpus/space.txt`, `data/corpus/vector-databases.pdf`
- Modify: `use_cases/rag_chatbot.py` (retrieve from chunks)
- Modify: `repl.py` (ingest corpus instead of indexing samples)

**Interfaces:**
- Consumes: `ingest_directory` (Task 5), `search_chunks` (Task 4), `build_pdf` (Task 3).
- Produces: `ChunkResult(id: int, content: str, source_path: str, chunk_index: int, score: float)` in `core.models`; `retrieve(query, store, embedder, top_k=3) -> list[ChunkResult]` and `generate_answer(query: str, context_chunks: list[ChunkResult]) -> str` in `use_cases.rag_chatbot` (`rag_chatbot.index_documents` is deleted; `repl.py` no longer imports it).

- [ ] **Step 1: Add the `ChunkResult` model to `core/models.py`**

Append:

```python
class ChunkResult(BaseModel):
    id: int
    content: str
    source_path: str
    chunk_index: int
    score: float
```

- [ ] **Step 2: Create the Markdown/text corpus files**

Create `data/corpus/machine-learning.md`:

```markdown
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.
Instead of hand-coding rules, a model discovers patterns in training data and
generalizes them to unseen inputs.

## Supervised and Unsupervised Learning

In supervised learning, the training data includes labeled examples: the model
sees inputs paired with correct outputs and learns the mapping between them.
Classification (predicting a category) and regression (predicting a number)
are the two classic supervised tasks.

Unsupervised learning works on unlabeled data. The model looks for structure
on its own — clustering similar items together or reducing high-dimensional
data to a compact representation. It is often used for exploratory analysis
and anomaly detection.

## Deep Learning

Deep learning uses neural networks with many layers to model complex patterns
in data. Each layer transforms its input into a slightly more abstract
representation, which lets deep networks learn features automatically instead
of relying on manual feature engineering. Deep learning powers modern image
recognition, speech processing, and large language models.
```

Create `data/corpus/photosynthesis.md`:

```markdown
# How Plants Make Food

Photosynthesis is the process by which green plants convert sunlight, water,
and carbon dioxide into glucose and oxygen. It takes place mainly in the
leaves, inside organelles called chloroplasts, which contain the green
pigment chlorophyll.

## The Light-Dependent Reactions

The first stage requires light. Chlorophyll absorbs sunlight and uses its
energy to split water molecules, releasing oxygen as a by-product. The
captured energy is stored temporarily in the molecules ATP and NADPH.

## The Calvin Cycle

The second stage does not need light directly. Using the ATP and NADPH from
the first stage, the plant fixes carbon dioxide from the air into glucose.
The glucose then fuels the plant's growth or is stored as starch for later
use.
```

Create `data/corpus/space.txt`:

```
Notes on physics and the solar system.

The speed of light in a vacuum is approximately 299,792 kilometers per
second, usually denoted as c in physics equations. Nothing that carries
information can travel faster than c; it is the universal speed limit.

Light from the Sun takes a little over eight minutes to reach Earth. The
Sun itself is an ordinary main-sequence star containing about 99.8 percent
of the solar system's total mass.

The solar system has eight planets. The four inner planets are small and
rocky, while the four outer planets are gas and ice giants. Beyond Neptune
lies the Kuiper belt, home of dwarf planets such as Pluto.
```

- [ ] **Step 3: Generate the corpus PDF**

```bash
python -c "from tests.helpers import build_pdf; open('data/corpus/vector-databases.pdf','wb').write(build_pdf('Vector databases index high-dimensional embeddings so that nearest-neighbor similarity search stays fast even with millions of items.'))"
```

Expected: `data/corpus/vector-databases.pdf` exists; verify with
`python -c "from ingestion.loaders import load_file; from pathlib import Path; print(load_file(Path('data/corpus/vector-databases.pdf')).text[:80])"` → prints the sentence start.

- [ ] **Step 4: Rewrite `use_cases/rag_chatbot.py`**

Replace the whole file with:

```python
"""
Use case: RAG Chatbot
----------------------
Ingest the local corpus (chunked + embedded), retrieve the most relevant
chunks from pgvector, and pass them as context to a local Ollama LLM to
generate a grounded answer.
"""

import os

import ollama
from dotenv import load_dotenv

from core.db import VectorStore
from core.embedder import Embedder
from core.models import ChunkResult
from ingestion.ingest import ingest_directory

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
CORPUS_DIR = "data/corpus"


def retrieve(
    query: str, store: VectorStore, embedder: Embedder, top_k: int = 3
) -> list[ChunkResult]:
    embedding = embedder.embed(query)
    rows = store.search_chunks(embedding, top_k=top_k)
    return [ChunkResult(**r) for r in rows]


def generate_answer(query: str, context_chunks: list[ChunkResult]) -> str:
    context = "\n\n".join(
        f"[Source: {c.source_path}#chunk{c.chunk_index}]\n{c.content}"
        for c in context_chunks
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know based on the provided documents."

        Context:
        {context}

        Question: {query}
        Answer:
    """

    client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    response = client.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"].strip()


def chat(query: str, store: VectorStore, embedder: Embedder) -> str:
    chunks = retrieve(query, store, embedder)
    return generate_answer(query, chunks)


def run(store: VectorStore, embedder: Embedder) -> None:
    print("\n=== RAG Chatbot ===")
    stats = ingest_directory(CORPUS_DIR, store, embedder)
    print(
        f"Corpus ready: {stats.ingested} ingested, {stats.skipped} unchanged, "
        f"{stats.deleted} deleted, {stats.failed} failed."
    )

    questions = [
        "What is machine learning?",
        "How do plants make food?",
        "What is the speed of light?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = chat(q, store, embedder)
        print(f"Answer:   {answer}")
```

(The grounding prompt is unchanged, so `evals/rag_chatbot/promptfooconfig.yaml` still mirrors it.)

- [ ] **Step 5: Update `repl.py`**

Replace the imports and the corpus-preparation lines:

```python
from core.db import VectorStore
from core.embedder import Embedder
from ingestion.ingest import ingest_directory
from use_cases.rag_chatbot import CORPUS_DIR, generate_answer, retrieve
```

and in `main()`, replace `index_documents(SAMPLE_DOCUMENTS, store, embedder)` with:

```python
        stats = ingest_directory(CORPUS_DIR, store, embedder)
        print(f"Corpus ready: {stats.ingested} ingested, {stats.skipped} unchanged.")
```

and in the sources display loop, replace `({d.source})` with `({d.source_path}#chunk{d.chunk_index})`.

Also update the module docstring's first line to: `Interactive RAG REPL — ask questions against the local corpus (data/corpus).`

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: everything PASSES (nothing imported `rag_chatbot.index_documents` except `repl.py`, which was updated).

- [ ] **Step 7: Manual end-to-end verification (needs Ollama running)**

```bash
docker compose up -d
python demo.py ingest data/corpus
```

Expected: `Ingested: 4  skipped (unchanged): 0  deleted: 0  failed: 0` (first run; 4 = 3 text files + 1 PDF).

```bash
python demo.py ingest data/corpus
```

Expected: `Ingested: 0  skipped (unchanged): 4 ...` — idempotency demonstrated.

```bash
python demo.py rag
```

Expected: three grounded answers (ML definition, photosynthesis, 299,792 km/s), no refusals.

- [ ] **Step 8: Commit**

```bash
git add core/models.py data/corpus use_cases/rag_chatbot.py repl.py
git commit -m "feat: RAG chatbot retrieves from chunked corpus with source citations

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Documentation + final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Update `README.md`**

Make these edits:

1. In the use-case table, add a row above the existing ones:

```markdown
| `python demo.py ingest [dir]` | Ingest a folder of .md/.txt/.pdf into chunked, embedded storage (default `data/corpus`) |
```

2. Under **Stack**, add:

```markdown
- **Ingestion**: recursive token-aware chunking (tiktoken `cl100k_base`, 600-token chunks, 80-token overlap), idempotent re-ingest via SHA-256 content hashes
```

3. In **Project Structure**, add after the `core/` block:

```markdown
ingestion/
  loaders.py     # file -> text (.md/.txt/.pdf)
  chunker.py     # text -> token-sized chunks with overlap
  ingest.py      # hash-diff orchestration: load -> chunk -> embed -> upsert
data/corpus/     # sample corpus ingested by the RAG chatbot
tests/           # pytest suite (unit + `-m integration`)
```

4. In **Key Concepts**, replace the RAG Chatbot paragraph with:

```markdown
**RAG Chatbot**: ingests `data/corpus` (chunked + embedded, skipping unchanged files by content hash), retrieves the most relevant chunks, then passes them with source citations as context to Ollama — the LLM only answers from retrieved context, not its own knowledge.
```

5. Add a **Testing** section before **Key Concepts**:

```markdown
## Testing

```bash
pytest                    # unit tests (no infra needed)
pytest -m integration     # DB tests — needs `docker compose up -d` (no Ollama needed; tests use a fake embedder)
```
```

- [ ] **Step 2: Full verification**

```bash
pytest -v
python demo.py ingest data/corpus
```

Expected: all tests pass; ingest reports all files skipped (unchanged).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document ingestion pipeline, corpus, and test suite

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
