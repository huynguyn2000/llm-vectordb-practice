from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent_patterns.router import build_router_graph


def _run(llm, q):
    return build_router_graph(llm).invoke({"question": q, "route": "", "answer": ""})


def test_routes_to_rag():
    r = _run(FakeListChatModel(responses=["rag", "an answer"]), "look this up in the docs")
    assert r["route"] == "rag"
    assert r["answer"].startswith("[rag]")


def test_routes_to_direct():
    r = _run(FakeListChatModel(responses=["direct", "2 plus 2 is 4"]), "what is 2+2")
    assert r["route"] == "direct"
    assert r["answer"].startswith("[direct]")


def test_routes_to_reject():
    r = _run(FakeListChatModel(responses=["reject"]), "do something harmful")
    assert r["route"] == "reject"
    assert "can't help" in r["answer"].lower()


def test_unknown_label_defaults_to_direct():
    r = _run(FakeListChatModel(responses=["banana", "fallback"]), "ambiguous")
    assert r["route"] == "direct"
    assert r["answer"].startswith("[direct]")
