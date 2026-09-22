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
