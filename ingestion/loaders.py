"""File -> text loaders. One function per format, dispatched by extension."""

from pathlib import Path

from pypdf import PdfReader

from core.models import RawDocument

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


class LoaderError(Exception):
    """A file could not be loaded (unsupported type or unreadable content)."""


def load_file(path: Path) -> RawDocument:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        try:
            reader = PdfReader(path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise LoaderError(f"cannot read PDF {path}: {exc}") from exc
    else:
        raise LoaderError(f"unsupported file type: {path}")
    return RawDocument(path=str(path), text=text)
