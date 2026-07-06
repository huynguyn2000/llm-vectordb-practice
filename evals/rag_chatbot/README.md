# RAG chatbot prompt eval

Evaluates the grounding prompt in `use_cases/rag_chatbot.py` (`generate_answer`)
for **faithfulness** (answers only from context) and **refusal** (says
"I don't know based on the provided documents." when the answer isn't present).

## Setup

```bash
# 1. Ollama running with the model the app uses
ollama serve            # if not already running
ollama pull llama3.2

# 2. API key for the LLM-judge (grades llm-rubric assertions)
export ANTHROPIC_API_KEY=sk-...

# 3. If Ollama isn't on the default host:
# export OLLAMA_BASE_URL=http://localhost:11434
```

## Run

```bash
cd evals/rag_chatbot
npx promptfoo@latest eval     # runs every test case
npx promptfoo@latest view     # opens the results grid in the browser
```

`eval` exits non-zero if any assertion fails, so it drops into CI as-is.

## What's tested

| Case | Checks |
|------|--------|
| Answerable (ML) | Correct, grounded answer; no false refusal |
| Unanswerable (speed of light, biology context) | Exact refusal phrase |
| Insufficient context | No hallucinated name/date; refuses |
| Multi-source | Synthesizes only relevant source |
| Context vs. prior knowledge | Trusts context over training data |
