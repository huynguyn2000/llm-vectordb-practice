import shutil

import pytest

pytest.importorskip("duckdb")
if shutil.which("dbt") is None:
    pytest.skip("dbt CLI not installed (de extra)", allow_module_level=True)

from de.pipeline import export_documents, run_dbt


def test_dbt_build_produces_documents():
    run_dbt()
    rows = export_documents()
    assert len(rows) >= 8  # one per seed row
    assert all(r["content"].strip() for r in rows)
    assert all(str(r["source"]).startswith("faq:") for r in rows)
    # content is question + answer text
    assert any("vector database" in r["content"].lower() for r in rows)
