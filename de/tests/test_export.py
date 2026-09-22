import pytest

duckdb = pytest.importorskip("duckdb")

from de.pipeline import export_documents, write_corpus


def test_export_and_write(tmp_path):
    dbpath = tmp_path / "w.duckdb"
    con = duckdb.connect(str(dbpath))
    con.execute("CREATE TABLE documents (id INTEGER, source VARCHAR, content VARCHAR)")
    con.execute(
        "INSERT INTO documents VALUES (1, 'faq:1', 'Q one\n\nA one'), (2, 'faq:2', 'Q two\n\nA two')"
    )
    con.close()

    rows = export_documents(dbpath)
    assert len(rows) == 2
    assert rows[0] == {"id": 1, "source": "faq:1", "content": "Q one\n\nA one"}

    n = write_corpus(rows, out_dir=str(tmp_path / "corpus"))
    assert n == 2
    assert (tmp_path / "corpus" / "faq_1.md").read_text() == "Q one\n\nA one"
