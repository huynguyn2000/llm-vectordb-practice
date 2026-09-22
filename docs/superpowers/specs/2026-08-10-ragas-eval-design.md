# Ragas Evaluation — Design

**Date:** 2026-08-10
**Status:** Approved (decisions locked via brainstorm; autonomous execution under session goal)
**Context:** Adds a Ragas evaluation harness that measures the RAG pipeline's
quality (faithfulness, relevance, retrieval precision/recall) over a hand-labeled
Q&A set, and **compares vector-only vs hybrid retrieval** side by side. This is
the measurement layer that lets every later change (Docling, vector-DB swaps,
etc.) be justified with numbers. Augments — does not replace — the existing
promptfoo prompt-level eval.

## Goals

- A hand-labeled dataset (~10–12 Q&A with ground-truth answers) over `data/corpus`.
- Compute the full Ragas metric suite: **faithfulness, answer_relevancy,
  context_precision, context_recall**.
- **Compare vector-only vs hybrid** retrieval on the same dataset, tabulated.
- Judge defaults to **local Ollama** ($0), overridable to an API judge via env.
- A standalone runner (`evals/ragas/run_eval.py`) — not a pytest gate.
- Reuse the existing pipeline (`use_cases.rag_chatbot`); no changes to
  `core/`, `ingestion/`, `search/`, `use_cases/`.

## Non-goals

- Ragas synthetic testset generation (poor quality with a local judge).
- Pytest score-threshold gating (noisy local-judge scores → flaky CI).
- Replacing promptfoo.

## Architecture & layout

```
evals/ragas/
  __init__.py
  dataset.json        # [{ "question": ..., "ground_truth": ... }, ...] ~10–12 rows
  judge.py            # build_judge() -> (llm, embeddings) for Ragas; local default, env override
  run_eval.py         # runs both retrieval configs, scores with Ragas, prints table + writes results.json
  tests/
    __init__.py
    test_dataset.py   # no-infra: dataset well-formed + loader shape
pyproject.toml        # + `eval` extra: ragas, datasets, langchain-ollama (langchain-openai/anthropic optional)
README.md             # + Evaluation (Ragas) section
```

## Dataset (`dataset.json`)

~10–12 questions grounded in `data/corpus` (machine-learning, photosynthesis,
space/physics, vector-databases), each with a concise human-written
`ground_truth`. Mostly answerable (so `context_recall` is meaningful), with a
couple whose answer spans/needs specific chunks. Example row:

```json
{ "question": "What is machine learning?",
  "ground_truth": "Machine learning is a subset of AI where systems learn patterns from data instead of being explicitly programmed." }
```

Loader `load_dataset(path="evals/ragas/dataset.json") -> list[dict]` validates
each row has non-empty `question` and `ground_truth`.

## Judge (`judge.py`)

`build_judge()` returns a `(llm, embeddings)` pair wrapped for Ragas
(`ragas.llms.LangchainLLMWrapper`, `ragas.embeddings.LangchainEmbeddingsWrapper`):

- **Default (local, $0):** `ChatOllama` + `OllamaEmbeddings` (`langchain-ollama`),
  models/host from the existing env (`LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_BASE_URL`).
- **Override:** env `RAGAS_JUDGE=openai|anthropic` selects `ChatOpenAI` /
  `ChatAnthropic` (+ their embeddings for openai; keep Ollama embeddings for
  anthropic) using the provider's API key. Documented; off by default.

The README notes local-judge scores are noisier/relative, not authoritative.

## Runner (`run_eval.py`)

For each config in `{"vector": retrieve_vector, "hybrid": retrieve}`:
1. For each dataset row: `chunks = config_fn(question, store, embedder, top_k=3)`;
   `answer = generate_answer(question, chunks)`; `contexts = [c.content for c in chunks]`.
2. Assemble a Ragas `EvaluationDataset` (columns: `user_input`, `response`,
   `retrieved_contexts`, `reference`).
3. `ragas.evaluate(dataset, metrics=[faithfulness, answer_relevancy,
   context_precision, context_recall], llm=judge_llm, embeddings=judge_emb)`.

Then print a side-by-side table (metric × {vector, hybrid}) and write
`evals/ragas/results.json` with per-config scores + per-row detail. Requires
Postgres + Ollama up (and the corpus ingested). CLI:
`python -m evals.ragas.run_eval` (optional `--top-k`, `--limit` for a quick
subset).

## Error handling

- Corpus empty / not ingested → runner prints a clear "ingest data/corpus first"
  message and exits non-zero.
- Ollama/Postgres down → error propagates with context.
- A single row erroring in generation → logged, skipped, counted; the run
  continues (one bad row shouldn't sink the whole eval).

## Testing

- `test_dataset.py` (no infra): `load_dataset()` returns ≥10 rows, every row has
  non-empty `question` + `ground_truth`, and JSON parses. Validates the loader
  and the labeled data — not the (slow, LLM-driven) eval itself.
- The eval run is verified manually (the runner prints the comparison table).
- Existing suite untouched.

## Cost & resources

- **$0** by default (local Ollama judge). API override incurs small per-run cost
  and sends eval questions + retrieved chunks to that provider.
- **Slow:** ragas makes several LLM calls per row per metric, ×2 configs — a
  ~12-row run is minutes on local llama3.2. `--limit` supports quick smoke runs.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Judge | Local Ollama default, API override via env | $0/local-consistent; credible numbers on demand |
| Dataset | Hand-labeled Q&A + ground truth (~10–12) | Unlocks full metric suite incl. context_recall |
| Metrics | faithfulness, answer_relevancy, context_precision, context_recall | Standard Ragas RAG quad |
| Scope | Compare vector-only vs hybrid | Proves the hybrid milestone; reusable template for later changes |
| Run | Standalone script, not a pytest gate | Slow + noisy local judge → unfit for CI gating |
| Reuse | `use_cases.rag_chatbot` retrieve/retrieve_vector/generate_answer | No pipeline changes; evaluates the real thing |
