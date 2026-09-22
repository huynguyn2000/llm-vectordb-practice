"""Run the labeled dataset through vector-only and hybrid retrieval, score each
with Ragas, and print a side-by-side comparison. Requires Postgres + Ollama up
and the corpus ingested (`python demo.py ingest data/corpus`).

Usage: python -m evals.ragas.run_eval [--top-k N] [--limit N]
"""

import argparse
import json
import sys

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from core.db import VectorStore
from core.embedder import Embedder
from evals.ragas.dataset_loader import load_eval_dataset
from evals.ragas.judge import build_judge
from use_cases.rag_chatbot import generate_answer, retrieve, retrieve_vector

CONFIGS = {"vector": retrieve_vector, "hybrid": retrieve}
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
RESULTS_PATH = "evals/ragas/results.json"


def _build_samples(rows, retrieve_fn, store, embedder, top_k):
    samples = []
    for row in rows:
        q = row["question"]
        try:
            chunks = retrieve_fn(q, store, embedder, top_k=top_k)
            answer = generate_answer(q, chunks)
        except Exception as exc:  # one bad row shouldn't sink the run
            print(f"WARNING: skipping '{q[:50]}...': {exc}")
            continue
        samples.append(
            {
                "user_input": q,
                "response": answer,
                "retrieved_contexts": [c.content for c in chunks],
                "reference": row["ground_truth"],
            }
        )
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    args = ap.parse_args()

    rows = load_eval_dataset()
    if args.limit:
        rows = rows[: args.limit]

    llm, emb = build_judge()
    results = {}

    with VectorStore() as store:
        embedder = Embedder()
        # Fail fast if the corpus isn't ingested.
        if not retrieve("machine learning", store, embedder, top_k=1):
            print("No chunks found — ingest the corpus first: python demo.py ingest data/corpus")
            sys.exit(1)

        for name, fn in CONFIGS.items():
            print(f"\n=== scoring config: {name} ({len(rows)} rows) ===")
            samples = _build_samples(rows, fn, store, embedder, args.top_k)
            dataset = EvaluationDataset.from_list(samples)
            scores = evaluate(dataset=dataset, metrics=METRICS, llm=llm, embeddings=emb)
            results[name] = {"scores": scores.to_pandas().mean(numeric_only=True).to_dict()}

    # Comparison table
    metric_names = [m.name for m in METRICS]
    print(f"\n{'metric':<22}{'vector':>10}{'hybrid':>10}")
    for m in metric_names:
        v = results["vector"]["scores"].get(m)
        h = results["hybrid"]["scores"].get(m)
        vs = f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        hs = f"{h:.3f}" if isinstance(h, (int, float)) else str(h)
        print(f"{m:<22}{vs:>10}{hs:>10}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
