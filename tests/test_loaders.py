import pytest

from ingestion.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_file
from tests.helpers import build_pdf


def test_supported_extensions():
    assert SUPPORTED_EXTENSIONS == {".md", ".txt", ".pdf"}


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


def test_load_pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf("Hello from a PDF document."))
    assert "Hello from a PDF document." in load_file(p).text


def test_corrupt_pdf_raises_loader_error(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf")
    with pytest.raises(LoaderError):
        load_file(p)


def test_unsupported_extension_raises_loader_error(tmp_path):
    p = tmp_path / "doc.docx"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(LoaderError):
        load_file(p)
