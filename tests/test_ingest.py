import pytest

from ingestion.ingest import ingest_directory
from tests.helpers import FakeEmbedder, build_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "zz-test-notes.md").write_text(
        "# Notes\n\nSome markdown notes about testing.", encoding="utf-8"
    )
    (tmp_path / "zz-test-plain.txt").write_text(
        "Plain text content for the ingest test.", encoding="utf-8"
    )
    (tmp_path / "zz-test-doc.pdf").write_bytes(build_pdf("Hello from a test PDF."))
    (tmp_path / "ignored.docx").write_text("should be ignored", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def clean_store(store):
    def _purge():
        for path in list(store.get_source_hashes()):
            if path.startswith("zz-test"):
                store.delete_source(path)

    _purge()
    yield store
    _purge()


def test_first_ingest_ingests_all_supported_files(store, corpus):
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.ingested == 3
    assert stats.skipped == stats.deleted == stats.failed == 0


def test_second_ingest_skips_everything(store, corpus):
    ingest_directory(corpus, store, FakeEmbedder())
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.skipped == 3
    assert stats.ingested == stats.deleted == stats.failed == 0


def test_modified_file_is_reingested(store, corpus):
    ingest_directory(corpus, store, FakeEmbedder())
    (corpus / "zz-test-notes.md").write_text(
        "# Notes\n\nEdited content.", encoding="utf-8"
    )
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.ingested == 1
    assert stats.skipped == 2


def test_removed_file_is_cleaned_up(store, corpus):
    ingest_directory(corpus, store, FakeEmbedder())
    (corpus / "zz-test-plain.txt").unlink()
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.deleted == 1
    assert stats.skipped == 2
    assert "zz-test-plain.txt" not in store.get_source_hashes()


def test_corrupt_pdf_is_counted_failed_and_run_continues(store, corpus):
    (corpus / "zz-test-broken.pdf").write_bytes(b"not a real pdf")
    stats = ingest_directory(corpus, store, FakeEmbedder())
    assert stats.failed == 1
    assert stats.ingested == 3
