import pytest

from core.db import VectorStore


@pytest.fixture
def store():
    try:
        s = VectorStore()
    except Exception:
        pytest.skip("Postgres not reachable - start it with `docker compose up -d`")
    yield s
    s.close()
