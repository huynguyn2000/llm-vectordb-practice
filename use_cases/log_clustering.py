"""
Use case: Log / Anomaly Clustering
------------------------------------
Embed log messages and use cosine distance in vector space to:
1. Find logs similar to a reference (query-based)
2. Detect anomalies — logs that are far from all others (outlier score)
"""

import numpy as np
from core.db import VectorStore
from core.embedder import Embedder
from core.models import Log, LogResult


def index_logs(logs: list[Log], store: VectorStore, embedder: Embedder) -> None:
    store.clear_logs()
    for log in logs:
        embedding = embedder.embed(log.message)
        store.insert_log(log.message, log.level, log.service, embedding)
    print(f"Indexed {len(logs)} log entries.")


def find_similar_logs(query: str, store: VectorStore, embedder: Embedder, top_k: int = 5) -> list[LogResult]:
    embedding = embedder.embed(query)
    rows = store.search_logs(embedding, top_k=top_k)
    return [LogResult(**r) for r in rows]


def detect_anomalies(store: VectorStore, top_n: int = 5) -> list[dict]:
    """
    Anomaly score = mean cosine distance to all other logs.
    Higher score = more isolated = more anomalous.
    """
    rows = store.get_all_logs_with_embeddings()
    if len(rows) < 2:
        return []

    ids = [r["id"] for r in rows]
    messages = [r["message"] for r in rows]
    services = [r["service"] for r in rows]
    embeddings = np.array([list(r["embedding"]) for r in rows], dtype=np.float32)

    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)
    sim_matrix = normed @ normed.T  # shape (N, N)

    # Mean distance to all others (exclude self on diagonal)
    np.fill_diagonal(sim_matrix, np.nan)
    mean_sim = np.nanmean(sim_matrix, axis=1)
    anomaly_scores = 1 - mean_sim  # high score = far from cluster

    ranked = sorted(
        zip(ids, messages, services, anomaly_scores.tolist()),
        key=lambda x: x[3],
        reverse=True,
    )
    return [
        {"id": id_, "message": msg, "service": svc, "anomaly_score": round(score, 4)}
        for id_, msg, svc, score in ranked[:top_n]
    ]


def run(store: VectorStore, embedder: Embedder) -> None:
    from data.logs import SAMPLE_LOGS

    print("\n=== Log / Anomaly Clustering ===")
    index_logs(SAMPLE_LOGS, store, embedder)

    print("\n-- Similar logs to 'database connection refused' --")
    results = find_similar_logs("database connection refused", store, embedder)
    for r in results:
        print(f"  [{r.score:.3f}] [{r.level}] {r.message}")

    print("\n-- Top anomalous log entries --")
    anomalies = detect_anomalies(store)
    for a in anomalies:
        print(f"  [score={a['anomaly_score']}] [{a['service']}] {a['message']}")
