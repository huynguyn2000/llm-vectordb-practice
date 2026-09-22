# Ragas Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Ragas eval harness that scores the RAG pipeline (faithfulness, answer_relevancy, context_precision, context_recall) over a hand-labeled Q&A set and compares vector-only vs hybrid retrieval, judged by a local Ollama model by default.

**Architecture:** New `evals/ragas/` package: `dataset.json` (labeled Q&A), `dataset_loader.py` (validated loader), `judge.py` (local/API judge factory for Ragas), `run_eval.py` (runs both retrieval configs through the existing `use_cases.rag_chatbot`, scores with Ragas, prints a comparison table + writes `results.json`). No changes to `core/`, `ingestion/`, `search/`, `use_cases/`.

**Tech Stack:** Python 3.11+, Ragas (`ragas`, `datasets`), `langchain-ollama`, the existing pgvector + Ollama RAG pipeline, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-ragas-eval-design.md`

## Global Constraints

- Judge defaults to **local Ollama** (`LLM_MODEL`/`EMBED_MODEL`/`OLLAMA_BASE_URL` env, default `llama3.2` / `nomic-embed-text` / `http://localhost:11434`); env `RAGAS_JUDGE=openai|anthropic` selects an API judge. No API key needed by default.
- Reuse `use_cases.rag_chatbot`: `retrieve` (hybrid) and `retrieve_vector` (vector-only) both `(query, store, embedder, top_k=3) -> list[ChunkResult]`; `generate_answer(query, chunks) -> str`. `ChunkResult.content` is the chunk text.
- Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
- Compare configs `{"vector": retrieve_vector, "hybrid": retrieve}` on the same dataset.
- Standalone runner (`python -m evals.ragas.run_eval`), NOT a pytest gate. The only pytest test is a no-infra dataset/loader check.
- No changes to `core/`, `ingestion/`, `search/`, `use_cases/`.
- **Ragas API is version-sensitive.** Pin `ragas>=0.2,<0.3`. The code below targets the 0.2.x API (`EvaluationDataset.from_list`, columns `user_input`/`response`/`retrieved_contexts`/`reference`, `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper`, metric singletons `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`). If the installed 0.2.x differs on metric import names or dataset columns, ADAPT to the installed version and note it in the report; STOP-and-report only if it can't be made to work.
- Run from repo root with the venv. Install with `pip install -e ".[dev,eval]"`.
- Every commit message ends with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Deps + dataset + loader + judge + dataset test

**Files:**
- Modify: `pyproject.toml` (add `eval` extra)
- Create: `evals/__init__.py` (empty), `evals/ragas/__init__.py` (empty)
- Create: `evals/ragas/dataset.json`
- Create: `evals/ragas/dataset_loader.py`
- Create: `evals/ragas/judge.py`
- Create: `evals/ragas/tests/__init__.py` (empty)
- Test: `evals/ragas/tests/test_dataset.py`

**Interfaces:**
- Produces: `load_eval_dataset(path: str = "evals/ragas/dataset.json") -> list[dict]` in `evals.ragas.dataset_loader`; `build_judge() -> tuple` (a `(llm, embeddings)` pair of Ragas wrappers) in `evals.ragas.judge`. Task 2 consumes both.

- [ ] **Step 1: Add the `eval` extra to `pyproject.toml`**

Under `[project.optional-dependencies]`:

```toml
eval = ["ragas>=0.2,<0.3", "datasets>=2.0", "langchain-ollama>=0.2"]
```

- [ ] **Step 2: Install**

```bash
pip install -e ".[dev,eval]"
```

Expected: ragas + datasets install. Record the installed ragas version in the report.

- [ ] **Step 3: Create package markers**

```bash
mkdir -p evals/ragas/tests && touch evals/__init__.py evals/ragas/__init__.py evals/ragas/tests/__init__.py
```

- [ ] **Step 4: Create `evals/ragas/dataset.json`**

Twelve labeled rows grounded in `data/corpus` (machine-learning.md, photosynthesis.md, space.txt, vector-databases.pdf):

```json
[
  {"question": "What is machine learning?", "ground_truth": "Machine learning is a subset of artificial intelligence where systems learn patterns from data and improve from experience instead of being explicitly programmed."},
  {"question": "What is the difference between supervised and unsupervised learning?", "ground_truth": "Supervised learning trains on labeled input-output pairs (e.g. classification, regression); unsupervised learning finds structure in unlabeled data, such as clustering or dimensionality reduction."},
  {"question": "What is deep learning?", "ground_truth": "Deep learning uses neural networks with many layers, each transforming its input into a more abstract representation, so features are learned automatically."},
  {"question": "How do plants make food?", "ground_truth": "Through photosynthesis: in chloroplasts, plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
  {"question": "What happens in the light-dependent reactions of photosynthesis?", "ground_truth": "Chlorophyll absorbs sunlight and uses it to split water, releasing oxygen, and stores energy in ATP and NADPH."},
  {"question": "What is the Calvin cycle?", "ground_truth": "The light-independent stage of photosynthesis that uses ATP and NADPH to fix carbon dioxide into glucose."},
  {"question": "What is the speed of light?", "ground_truth": "About 299,792 kilometers per second in a vacuum, denoted c, the universal speed limit for information."},
  {"question": "How long does sunlight take to reach Earth?", "ground_truth": "A little over eight minutes."},
  {"question": "How many planets are in the solar system and how are they grouped?", "ground_truth": "Eight planets: four small rocky inner planets and four gas and ice giant outer planets."},
  {"question": "What are vector databases used for?", "ground_truth": "They index high-dimensional embeddings so nearest-neighbor similarity search stays fast even across millions of items."},
  {"question": "What is chlorophyll?", "ground_truth": "The green pigment in chloroplasts that absorbs sunlight to drive photosynthesis."},
  {"question": "Why can nothing travel faster than light?", "ground_truth": "The speed of light in a vacuum is the universal speed limit; nothing carrying information can exceed it."}
]
```

- [ ] **Step 5: Create the loader `evals/ragas/dataset_loader.py`**

```python
"""Load and validate the hand-labeled Ragas eval dataset."""

import json
from pathlib import Path

DEFAULT_PATH = "evals/ragas/dataset.json"


def load_eval_dataset(path: str = DEFAULT_PATH) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    for i, row in enumerate(rows):
        if not row.get("question", "").strip():
            raise ValueError(f"{path} row {i}: empty 'question'")
        if not row.get("ground_truth", "").strip():
            raise ValueError(f"{path} row {i}: empty 'ground_truth'")
    return rows
```

- [ ] **Step 6: Create the judge factory `evals/ragas/judge.py`**

```python
"""Judge model factory for Ragas. Local Ollama by default; API override via
RAGAS_JUDGE=openai|anthropic. Returns Ragas-wrapped (llm, embeddings)."""

import os

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


def build_judge():
    provider = os.getenv("RAGAS_JUDGE", "ollama").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        emb = OpenAIEmbeddings(model=os.getenv("RAGAS_OPENAI_EMBED", "text-embedding-3-small"))
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from langchain_ollama import OllamaEmbeddings

        llm = ChatAnthropic(model=os.getenv("RAGAS_ANTHROPIC_MODEL", "claude-sonnet-4-5"), temperature=0)
        emb = OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:  # local Ollama (default, $0)
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=os.getenv("LLM_MODEL", "llama3.2"), base_url=base, temperature=0)
        emb = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"), base_url=base)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)
```

- [ ] **Step 7: Write the no-infra dataset test `evals/ragas/tests/test_dataset.py`**

```python
import pytest

from evals.ragas.dataset_loader import load_eval_dataset


def test_dataset_loads_and_is_well_formed():
    rows = load_eval_dataset()
    assert len(rows) >= 10
    assert all(r["question"].strip() and r["ground_truth"].strip() for r in rows)


def test_loader_rejects_empty_ground_truth(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('[{"question": "q", "ground_truth": ""}]', encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_dataset(str(bad))
```

- [ ] **Step 8: Run the test**

Run: `.venv/bin/pytest evals/ragas/tests/test_dataset.py -v`
Expected: 2 pass (no infra). Also confirm `.venv/bin/pytest --collect-only -q` still collects the whole suite with no errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml evals/__init__.py evals/ragas/
git commit -m "feat: Ragas eval dataset, loader, and judge factory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Comparison runner

**Files:**
- Create: `evals/ragas/run_eval.py`

**Interfaces:**
- Consumes: `load_eval_dataset`, `build_judge` (Task 1); `use_cases.rag_chatbot.retrieve`/`retrieve_vector`/`generate_answer`; `core.db.VectorStore`, `core.embedder.Embedder`.
- Produces: `python -m evals.ragas.run_eval` → comparison table + `evals/ragas/results.json`.

- [ ] **Step 1: Implement `evals/ragas/run_eval.py`**

```python
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
            results[name] = {"scores": scores._repr_dict if hasattr(scores, "_repr_dict") else dict(scores)}

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
```

Note: `evaluate(...)` returns a Ragas `EvaluationResult`; extracting the
aggregate per-metric scores is version-sensitive. If `results[name]["scores"]`
doesn't expose a `{metric_name: float}` mapping on the installed 0.2.x, adapt the
extraction (e.g. `scores.to_pandas().mean(numeric_only=True).to_dict()`) so the
comparison table prints per-metric floats. Note the adaptation in the report.

- [ ] **Step 2: Manual smoke run (needs Postgres + Ollama + corpus)**

```bash
docker compose up -d
.venv/bin/python demo.py ingest data/corpus
.venv/bin/python -m evals.ragas.run_eval --limit 2
```

Expected: prints a `metric | vector | hybrid` table with numeric scores for the
4 metrics and writes `results.json`. `--limit 2` keeps the local-judge run to a
couple of minutes. Capture the table in the report. (If a metric shows `nan`
with the local judge, that's acceptable/expected noise — note it; the harness
working is the deliverable, not the score quality.)

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (the new no-infra dataset test included), no collection errors.

- [ ] **Step 4: Commit**

```bash
git add evals/ragas/run_eval.py evals/ragas/results.json
git commit -m "feat: Ragas comparison runner (vector-only vs hybrid)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Docs + final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Update `README.md`**

1. Under **Stack**, add:

```markdown
- **Evaluation**: Ragas — faithfulness / answer-relevancy / context-precision / context-recall over a labeled Q&A set, comparing vector-only vs hybrid (local Ollama judge by default)
```

2. In **Project Structure**, add:

```markdown
evals/ragas/
  dataset.json        # hand-labeled Q&A (question + ground truth)
  judge.py            # local-Ollama (default) / API judge factory for Ragas
  run_eval.py         # scores vector-only vs hybrid, writes results.json
```

3. Add an **Evaluation (Ragas)** section before **Key Concepts**:

````markdown
## Evaluation (Ragas)

Measure retrieval + generation quality and compare retrieval modes:

```bash
docker compose up -d
pip install -e ".[eval]"
python demo.py ingest data/corpus
python -m evals.ragas.run_eval            # add --limit 2 for a quick run
```

It scores faithfulness, answer-relevancy, context-precision, and context-recall
for **vector-only vs hybrid** retrieval over `evals/ragas/dataset.json`, prints a
side-by-side table, and writes `results.json`. The judge defaults to local
`llama3.2` ($0) — set `RAGAS_JUDGE=openai` (or `anthropic`) with the provider's
API key for more stable scores. Local-judge scores are noisy — read them as
relative, not authoritative.
````

- [ ] **Step 2: Final verification**

Run: `.venv/bin/pytest -q`
Expected: full suite passes, no collection errors.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document Ragas evaluation harness

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
