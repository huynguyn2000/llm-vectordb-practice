"""Dagster code location: assets, the asset check, a materialization job, a
daily schedule, and the resources they need."""

from dagster import Definitions, ScheduleDefinition, define_asset_job

from orchestration.assets import chunks_present, corpus_source, pgvector_chunks
from orchestration.resources import EmbedderResource, VectorStoreResource

ingest_job = define_asset_job(name="ingest_corpus_job", selection="*")

daily_schedule = ScheduleDefinition(
    name="daily_ingest",
    job=ingest_job,
    cron_schedule="0 6 * * *",
)

defs = Definitions(
    assets=[corpus_source, pgvector_chunks],
    asset_checks=[chunks_present],
    jobs=[ingest_job],
    schedules=[daily_schedule],
    resources={
        "vector_store": VectorStoreResource(),
        "embedder": EmbedderResource(),
    },
)
