"""Re-export the shared `store` fixture into this sibling test dir. Importing a
fixture into a conftest's namespace shares it without the deprecated
`pytest_plugins` mechanism (which errors in a non-root conftest on pytest 9)."""

from tests.conftest import store  # noqa: F401
