"""Bridge fixture scope: orchestration/tests/ is a sibling of tests/, not a
descendant, so pytest won't pick up tests/conftest.py's `store` fixture
automatically. Re-export it directly so test_materialize.py can depend on
it — pytest discovers fixture-decorated callables by introspecting a
conftest's namespace, so a plain import is enough (and, unlike
`pytest_plugins` in a non-root conftest, doesn't break whole-suite
collection)."""

from tests.conftest import store  # noqa: F401
