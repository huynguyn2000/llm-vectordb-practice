"""The ingestion pipeline declared as Dagster assets with lineage
(corpus_source -> pgvector_chunks) plus a post-condition asset check.

The corpus directory is read from the CORPUS_DIR env var at runtime (default
data/corpus) so tests can retarget it without touching the real corpus."""

import os
from pathlib import Path

from dagster import (
    AssetCheckResult,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from ingestion.ingest import ingest_directory
from ingestion.loaders import SUPPORTED_EXTENSIONS
from orchestration.resources import EmbedderResource, VectorStoreResource


def _corpus_dir() -> str:
    return os.getenv("CORPUS_DIR", "data/corpus")


@asset
def corpus_source() -> MaterializeResult:
    """Validate the corpus directory has supported files; fail fast if empty."""
    root = Path(_corpus_dir())
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No supported files found in {root}")
    return MaterializeResult(
        metadata={
            "file_count": MetadataValue.int(len(files)),
            "corpus_dir": MetadataValue.text(str(root)),
        }
    )


@asset(deps=[corpus_source])
def pgvector_chunks(
    vector_store: VectorStoreResource, embedder: EmbedderResource
) -> MaterializeResult:
    """Ingest the corpus into pgvector (idempotent hash-diff). Emit IngestStats."""
    store = vector_store.get_store()
    try:
        stats = ingest_directory(_corpus_dir(), store, embedder.get_embedder())
    finally:
        store.close()
    return MaterializeResult(
        metadata={
            "ingested": MetadataValue.int(stats.ingested),
            "skipped": MetadataValue.int(stats.skipped),
            "deleted": MetadataValue.int(stats.deleted),
            "failed": MetadataValue.int(stats.failed),
        }
    )


@asset_check(asset=pgvector_chunks)
def chunks_present(vector_store: VectorStoreResource) -> AssetCheckResult:
    """Post-condition: pgvector holds at least one chunk."""
    store = vector_store.get_store()
    try:
        with store.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            count = cur.fetchone()[0]
    finally:
        store.close()
    return AssetCheckResult(
        passed=count > 0, metadata={"chunk_count": MetadataValue.int(count)}
    )
