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
