"""Populate a Chroma backend from a corpus directory, reusing the ingestion
loaders + chunker so it holds the same chunks as pgvector."""

from pathlib import Path

from ingestion.chunker import chunk_text
from ingestion.loaders import SUPPORTED_EXTENSIONS, load_file
from vectorstores.chroma import ChromaBackend


def build_chroma_backend_from_corpus(corpus_dir, embedder, backend=None) -> ChromaBackend:
    backend = backend or ChromaBackend.persistent()
    root = Path(corpus_dir)
    items, cid = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        doc = load_file(path)
        rel = str(path.relative_to(root))
        for ch in chunk_text(doc.text):
            items.append(
                {
                    "id": cid,
                    "content": ch.content,
                    "source_path": rel,
                    "chunk_index": ch.chunk_index,
                    "embedding": embedder.embed(ch.content),
                }
            )
            cid += 1
    if items:
        backend.upsert(items)
    return backend
