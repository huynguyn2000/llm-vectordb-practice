"""Bridge fixture scope: orchestration/tests/ is a sibling of tests/, not a
descendant, so pytest won't pick up tests/conftest.py's `store` fixture
automatically. Register it explicitly so test_materialize.py can depend on
it."""

pytest_plugins = ["tests.conftest"]
