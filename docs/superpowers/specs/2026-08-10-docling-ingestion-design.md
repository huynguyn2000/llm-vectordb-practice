# Docling Ingestion Upgrade — Design

**Date:** 2026-08-10
**Status:** Approved (autonomous execution under session goal)
**Context:** Upgrades document parsing in the ingestion pipeline from basic
`pypdf` text extraction to **Docling** (IBM's layout/table-aware converter →
Markdown), and adds `.docx` support. Docling is an **opt-in extra with a pypdf
fallback**, so the default install stays light and PDF ingest keeps working
without it. This raises retrieval quality at the source (better chunks) and sets
up a pypdf-vs-Docling comparison the Ragas harness can quantify.

## Goals

- Route `.pdf` and `.docx` through Docling → Markdown when Docling is installed.
- Keep the default install working: `.pdf` falls back to `pypdf` when Docling is
  absent; `pypdf` stays a base dependency.
- Add `.docx` to `SUPPORTED_EXTENSIONS` (requires Docling; clear error if absent).
- Fast unit tests that mock Docling (no multi-GB model download needed for CI).
- Only touch `ingestion/loaders.py` (+ its tests, pyproject, README). No changes
  to chunker/ingest/core/search.

## Non-goals

- Making Docling a base dependency (it pulls torch + downloads ~GB of layout
  models on first convert — too heavy for the default install).
- OCR tuning, `.pptx`/`.html` (Docling supports them; trivially addable later).
- Re-ingesting the corpus as part of this change.

## Architecture (`ingestion/loaders.py`)

```
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}

load_file(path):
  .md/.txt  -> path.read_text (unchanged)
  .pdf      -> _load_pdf(path): Docling→markdown if available, else pypdf fallback
  .docx     -> _docling_markdown(path): Docling→markdown; LoaderError if Docling absent
  else      -> LoaderError
```

Helpers:
- `_docling_available() -> bool` — lazy `importlib.util.find_spec("docling")`.
- `_docling_markdown(path) -> str` — lazily `from docling.document_converter import
  DocumentConverter`, `DocumentConverter().convert(path).document.export_to_markdown()`;
  wrap failures in `LoaderError`.
- `_load_pdf(path) -> str` — if `_docling_available()`: `_docling_markdown(path)`;
  else the existing `pypdf` extraction. Both wrap failures in `LoaderError`.

Docling is imported **lazily inside the helper**, so importing `loaders.py` and
the `.md`/`.txt`/pypdf paths never require Docling to be installed.

## Dependencies

- `pyproject.toml`: new `docling` extra = `["docling>=2.0"]`. `pypdf` stays in
  base deps (fallback). Install the upgrade with `pip install -e ".[docling]"`.

## Testing

Unit tests (no Docling install, no models) in `tests/test_loaders.py`:
- `SUPPORTED_EXTENSIONS == {".md", ".txt", ".pdf", ".docx"}`.
- `.md`/`.txt` load (unchanged).
- `.pdf` via the **pypdf fallback** still extracts text from the synthetic
  `build_pdf` fixture (tests the default/no-Docling path deterministically).
- Dispatch tests via monkeypatch: patch `_docling_markdown` to a stub and assert
  `.pdf` uses it when `_docling_available()` is True, and `.docx` routes to it;
  patch `_docling_available` to False and assert `.docx` raises `LoaderError`
  (Docling-absent message) and `.pdf` falls back to pypdf.
- Docling conversion error (stub raising) → `LoaderError`.
- Unsupported extension → `LoaderError`.

Manual/integration (with Docling installed): convert the real
`data/corpus/vector-databases.pdf` and confirm Markdown output. **If the Docling
model download is too heavy to complete in-session, document that** — the unit
tests (mocked) prove the dispatch/fallback logic; the real conversion is the
manual step. (Same honest posture as the Ragas local-judge caveat.)

## Error handling

- `.docx` with Docling absent → `LoaderError("install the docling extra: pip install -e '.[docling]'")`.
- Any Docling/pypdf conversion exception → `LoaderError` with the path + cause.
- `.md`/`.txt` unchanged.

## Cost & resources

- **$0**, but Docling is heavy: installing `.[docling]` pulls torch and, on the
  first PDF/DOCX convert, downloads ~GB of layout models (needs network once).
  The default install (no extra) stays light via the pypdf fallback.

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Docling delivery | Opt-in `docling` extra + pypdf fallback | Default install stays light + PDF keeps working; Docling is a measurable upgrade |
| Formats | Docling for `.pdf`/`.docx`; pypdf fallback for `.pdf` | Adds `.docx`; PDF quality upgrade without hard dependency |
| Import | Lazy inside helper | `loaders.py` importable without Docling; fast no-Docling tests |
| Tests | Mock Docling; pypdf-fallback path tested for real | CI doesn't need multi-GB models; dispatch/fallback still proven |
| pptx/html | Out of scope | YAGNI; trivially addable via the same converter |
