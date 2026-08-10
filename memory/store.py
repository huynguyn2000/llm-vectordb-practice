"""Thin Mem0 wrapper. mem0 is imported lazily so this module loads without the
`memory` extra installed."""

from memory.config import build_memory_config


def build_memory():
    """Construct a local Mem0 Memory. Raises a clear error if mem0ai is absent."""
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise ImportError(
            "mem0 is not installed — install the memory extra: pip install -e '.[memory]'"
        ) from exc
    return Memory.from_config(build_memory_config())


def remember(mem, user_id: str, messages) -> None:
    """Store messages (a string or a list of {role, content}) for a user."""
    mem.add(messages, user_id=user_id)


def recall(mem, user_id: str, query: str) -> list[str]:
    """Return memory strings relevant to the query for a user."""
    result = mem.search(query, user_id=user_id)
    # mem0 returns either {"results": [{"memory": ...}, ...]} or a list; handle both.
    rows = result.get("results", result) if isinstance(result, dict) else result
    return [r.get("memory", str(r)) if isinstance(r, dict) else str(r) for r in rows]
