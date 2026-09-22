# Docling Ingestion Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `.pdf`/`.docx` parsing to Docling (→ Markdown) as an opt-in extra with a pypdf fallback, keeping the default install light and tests fast.

**Architecture:** `ingestion/loaders.py` routes `.pdf`/`.docx` through a lazily-imported Docling helper when the `docling` extra is installed; `.pdf` falls back to pypdf otherwise; `.md`/`.txt` unchanged. Unit tests mock Docling so CI needs no model download.

**Tech Stack:** Python 3.11+, Docling (optional extra), pypdf (base fallback), pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-docling-ingestion-design.md`

## Global Constraints

- `SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}`.
- Docling imported **lazily inside a helper** — `loaders.py` must import fine without Docling installed; `.md`/`.txt`/pypdf paths never require it.
- `.pdf`: Docling when available, else pypdf fallback. `.docx`: Docling required → `LoaderError` (mentioning `pip install -e '.[docling]'`) if absent.
- All conversion failures wrap in `LoaderError`. `load_file(path: Path) -> RawDocument` signature unchanged.
- `pypdf` stays a base dependency; Docling goes in a new `docling` extra (`docling>=2.0`).
- Only `ingestion/loaders.py`, `tests/test_loaders.py`, `pyproject.toml`, `README.md` change. No changes to chunker/ingest/core/search.
- Unit tests must pass with NO Docling installed (mock the Docling helper; force pypdf path via monkeypatch).
- Every commit ends with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Docling loader + deps + tests

**Files:**
- Modify: `pyproject.toml` (docling extra)
- Modify: `ingestion/loaders.py`
- Modify: `tests/test_loaders.py`

**Interfaces:**
- Produces: `load_file(path) -> RawDocument` (unchanged signature); module-level `_docling_available() -> bool`, `_docling_markdown(path) -> str`, `_load_pdf(path) -> str` (tests monkeypatch these). `SUPPORTED_EXTENSIONS` gains `.docx`.

- [ ] **Step 1: Add the `docling` extra to `pyproject.toml`**

Under `[project.optional-dependencies]`:

```toml
docling = ["docling>=2.0"]
```

(Do NOT add docling to base deps; do NOT install it now — Task 1 tests must pass without it.)

- [ ] **Step 2: Rewrite `ingestion/loaders.py`**

```python
"""File -> text loaders. One function per format, dispatched by extension.

PDF and DOCX use Docling (layout/table-aware -> Markdown) when the optional
`docling` extra is installed; PDF falls back to pypdf otherwise."""

import importlib.util
from pathlib import Path

from pypdf import PdfReader

from core.models import RawDocument

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


class LoaderError(Exception):
    """A file could not be loaded (unsupported type or unreadable content)."""


def _docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def _docling_markdown(path: Path) -> str:
    """Convert a document to Markdown via Docling. LoaderError on failure."""
    try:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:
        raise LoaderError(f"Docling could not convert {path}: {exc}") from exc


def _load_pdf(path: Path) -> str:
    if _docling_available():
        return _docling_markdown(path)
    try:
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise LoaderError(f"cannot read PDF {path}: {exc}") from exc


def load_file(path: Path) -> RawDocument:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        text = _load_pdf(path)
    elif ext == ".docx":
        if not _docling_available():
            raise LoaderError(
                f"cannot read {path}: .docx needs the docling extra "
                "(pip install -e '.[docling]')"
            )
        text = _docling_markdown(path)
    else:
        raise LoaderError(f"unsupported file type: {path}")
    return RawDocument(path=str(path), text=text)
```

- [ ] **Step 3: Rewrite `tests/test_loaders.py`**

```python
import pytest

import ingestion.loaders as loaders
from ingestion.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_file
from tests.helpers import build_pdf


def test_supported_extensions():
    assert SUPPORTED_EXTENSIONS == {".md", ".txt", ".pdf", ".docx"}


def test_load_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody text.", encoding="utf-8")
    doc = load_file(p)
    assert doc.text == "# Title\n\nBody text."
    assert doc.path == str(p)


def test_load_txt(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Plain text.", encoding="utf-8")
    assert load_file(p).text == "Plain text."


def test_pdf_uses_pypdf_fallback_when_docling_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "_docling_available", lambda: False)
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf("Hello from a PDF document."))
    assert "Hello from a PDF document." in load_file(p).text


def test_pdf_uses_docling_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "_docling_available", lambda: True)
    monkeypatch.setattr(loaders, "_docling_markdown", lambda path: "# Docling MD")
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 stub")  # never parsed; docling is stubbed
    assert load_file(p).text == "# Docling MD"


def test_docx_uses_docling_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "_docling_available", lambda: True)
    monkeypatch.setattr(loaders, "_docling_markdown", lambda path: "docx as markdown")
    p = tmp_path / "doc.docx"
    p.write_bytes(b"stub")
    assert load_file(p).text == "docx as markdown"


def test_docx_without_docling_raises_loader_error(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "_docling_available", lambda: False)
    p = tmp_path / "doc.docx"
    p.write_bytes(b"stub")
    with pytest.raises(LoaderError):
        load_file(p)


def test_docling_conversion_error_raises_loader_error(tmp_path, monkeypatch):
    def boom(path):
        raise LoaderError("boom")

    monkeypatch.setattr(loaders, "_docling_available", lambda: True)
    monkeypatch.setattr(loaders, "_docling_markdown", boom)
    p = tmp_path / "doc.docx"
    p.write_bytes(b"stub")
    with pytest.raises(LoaderError):
        load_file(p)


def test_corrupt_pdf_raises_loader_error(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "_docling_available", lambda: False)
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf")
    with pytest.raises(LoaderError):
        load_file(p)


def test_unsupported_extension_raises_loader_error(tmp_path):
    p = tmp_path / "doc.rtf"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(LoaderError):
        load_file(p)
```

- [ ] **Step 4: Run the loader tests (no Docling installed)**

Run: `.venv/bin/pytest tests/test_loaders.py -v`
Expected: all pass WITHOUT docling installed (Docling paths are monkeypatched; pypdf fallback is real). Also `.venv/bin/pytest --collect-only -q` → no collection errors.

- [ ] **Step 5: Confirm `loaders.py` imports without Docling**

Run: `.venv/bin/python -c "import ingestion.loaders; print(ingestion.loaders._docling_available())"`
Expected: prints `False` (docling not installed), no ImportError.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml ingestion/loaders.py tests/test_loaders.py
git commit -m "feat: Docling PDF/DOCX loader (opt-in extra, pypdf fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Docs + real-conversion verify

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Update `README.md`**

1. Under **Stack**, update the Ingestion bullet (or add one):

```markdown
- **Document parsing**: Docling (layout/table-aware → Markdown) for `.pdf`/`.docx` via the `docling` extra; falls back to `pypdf` for PDFs when the extra isn't installed
```

2. In **Project Structure**, update the loaders line:

```markdown
  loaders.py     # file -> text (.md/.txt plain; .pdf/.docx via Docling, pypdf fallback)
```

3. Add a **Document parsing (Docling)** subsection near the ingestion docs:

````markdown
### Better parsing with Docling

By default, PDFs are parsed with `pypdf` (light, text-only). For layout- and
table-aware parsing (PDF/DOCX → Markdown), install the Docling extra:

```bash
pip install -e ".[docling]"     # pulls torch; first convert downloads ~GB of models
python demo.py ingest data/corpus
```

With the extra installed, `.pdf` and `.docx` are converted via Docling; without
it, `.pdf` still works via the pypdf fallback and `.docx` raises a clear error.
````

- [ ] **Step 2: Best-effort real Docling conversion (may be heavy)**

```bash
pip install -e ".[docling]"
timeout 600 .venv/bin/python -c "from pathlib import Path; import ingestion.loaders as L; print(L.load_file(Path('data/corpus/vector-databases.pdf')).text[:300])"
```

If it completes: confirm it prints Markdown-ish text from the PDF; capture a snippet in the report. **If the install or first-convert model download is too heavy/slow to finish in the timeout, that's acceptable** — note it in the report as the documented Docling-is-heavy caveat (the mocked unit tests already prove the dispatch/fallback logic). Do NOT block the task on it.

- [ ] **Step 3: Final verification**

Run: `.venv/bin/pytest tests/test_loaders.py -q` (and `.venv/bin/pytest -q` if docling didn't get installed / to confirm no regressions)
Expected: loader tests pass. (If docling got installed and a full-suite integration test now routes a synthetic PDF through real Docling and fails/slows, note it — the loader unit tests force the pypdf path via monkeypatch and remain the gate.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document Docling parsing upgrade

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
