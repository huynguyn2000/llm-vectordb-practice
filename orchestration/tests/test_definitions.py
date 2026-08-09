"""Load/structure tests — no Postgres or Ollama needed. Constructing the
Definitions and resolving the job validates the asset graph and that every
resource key an asset/check requests is provided; resources are lazy, so no
connection is opened here."""

from orchestration import definitions as d
from orchestration.assets import corpus_source, pgvector_chunks


def test_definitions_import_ok():
    assert d.defs is not None


def test_job_resolves():
    # Resolving the job builds the asset graph and checks resource requirements
    # are satisfied — raises if wiring is broken.
    assert d.defs.get_job_def("ingest_corpus_job") is not None


def test_assets_have_expected_keys():
    assert corpus_source.key.path[-1] == "corpus_source"
    assert pgvector_chunks.key.path[-1] == "pgvector_chunks"


def test_schedule_is_daily():
    assert d.daily_schedule.cron_schedule == "0 6 * * *"
    assert d.daily_schedule.name == "daily_ingest"
