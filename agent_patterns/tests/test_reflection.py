from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent_patterns.reflection import build_reflection_graph


def _run(llm, task, max_revisions=2):
    return build_reflection_graph(llm, max_revisions=max_revisions).invoke(
        {"task": task, "draft": "", "critique": "", "revisions": 0, "approved": False}
    )


def test_approved_first_pass():
    # generate -> "draft one"; reflect -> "APPROVE"
    r = _run(FakeListChatModel(responses=["draft one", "APPROVE"]), "task")
    assert r["approved"] is True
    assert r["revisions"] == 0
    assert r["draft"] == "draft one"


def test_one_revision_then_approve():
    # generate -> d1; reflect -> critique; revise -> d2; reflect -> APPROVE
    r = _run(FakeListChatModel(responses=["draft one", "improve X", "draft two", "APPROVE"]), "task")
    assert r["revisions"] == 1
    assert r["draft"] == "draft two"
    assert r["approved"] is True


def test_stops_at_max_revisions():
    # never approves: generate, reflect, revise, reflect, revise, reflect -> cap
    r = _run(FakeListChatModel(responses=["d0", "crit", "d1", "crit", "d2", "crit"]), "task", max_revisions=2)
    assert r["revisions"] == 2
    assert r["approved"] is False
