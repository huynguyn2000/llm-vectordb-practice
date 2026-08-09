import json
from pathlib import Path


def test_langgraph_json_is_valid_and_points_to_factory():
    cfg = json.loads(Path("langgraph.json").read_text())
    assert cfg["graphs"]["agentic_rag"] == "./langgraph_rag/studio.py:make_graph"
    assert cfg["dependencies"] == ["."]


def test_studio_factory_is_importable_and_callable():
    from langgraph_rag import studio
    assert callable(studio.make_graph)
