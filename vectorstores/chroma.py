"""ChromaDB-backed VectorBackend (embedded; cosine space). chromadb imported
lazily so this module loads without the optional extra."""


class ChromaBackend:
    name = "chroma"

    def __init__(self, collection):
        self._collection = collection

    @classmethod
    def _collection_for(cls, client, name: str):
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    @classmethod
    def in_memory(cls, name: str = "chunks") -> "ChromaBackend":
        import chromadb

        return cls(cls._collection_for(chromadb.EphemeralClient(), name))

    @classmethod
    def persistent(cls, path: str = ".chroma", name: str = "chunks") -> "ChromaBackend":
        import chromadb

        return cls(cls._collection_for(chromadb.PersistentClient(path=path), name))

    def upsert(self, items: list[dict]) -> None:
        self._collection.upsert(
            ids=[str(it["id"]) for it in items],
            documents=[it["content"] for it in items],
            embeddings=[it["embedding"] for it in items],
            metadatas=[
                {"source_path": it["source_path"], "chunk_index": it["chunk_index"]}
                for it in items
            ],
        )

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]
        out = []
        for i in range(len(ids)):
            raw_id = ids[i]
            # Clamp: chromadb's float32 cosine distance can be a hair below 0
            # for near-exact matches (e.g. -1.4e-6), which would otherwise
            # push the derived score just past the mathematical bound of 1.0.
            score = max(-1.0, min(1.0, 1.0 - dists[i]))
            out.append(
                {
                    "id": int(raw_id) if raw_id.isdigit() else raw_id,
                    "content": docs[i],
                    "source_path": metas[i].get("source_path"),
                    "chunk_index": metas[i].get("chunk_index"),
                    "score": score,
                }
            )
        return out
