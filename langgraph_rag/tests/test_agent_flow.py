from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from langgraph_rag.graph import build_graph

_DOC = Document(
    page_content="Machine learning learns patterns from data.",
    metadata={"source_path": "zz.md", "chunk_index": 0},
)


def _fixed_retriever():
    # Returns exactly one doc per call -> one grade LLM call per round.
    return RunnableLambda(lambda q: [_DOC])


def _run(graph, question):
    return graph.invoke(
        {"question": question, "documents": [], "generation": "", "attempts": 0}
    )


def test_happy_path_grade_yes_then_generate():
    llm = FakeListChatModel(responses=["yes", "ML learns from data."])
    result = _run(build_graph(_fixed_retriever(), llm), "what is machine learning")
    assert result["generation"] == "ML learns from data."
    assert result["attempts"] == 1


def test_retry_loop_grade_no_then_transform_then_generate():
    # round 1: grade "no" -> transform -> round 2: grade "yes" -> generate
    llm = FakeListChatModel(
        responses=["no", "rewritten question", "yes", "Looped answer."]
    )
    result = _run(build_graph(_fixed_retriever(), llm), "vague query")
    assert result["attempts"] == 2
    assert result["generation"] == "Looped answer."


def test_exhausted_retries_generates_with_empty_docs():
    # Grader always says "no": round1 grade(no)->transform->round2 grade(no)->
    # attempts hits max_attempts(2) so the decider routes to generate with
    # empty documents (the loop-termination cap branch).
    llm = FakeListChatModel(responses=["no", "rewritten question", "no", "final answer"])
    result = _run(build_graph(_fixed_retriever(), llm), "vague query")
    assert result["attempts"] == 2
    assert result["documents"] == []          # generate reached via the cap, not via relevant docs
    assert result["generation"] == "final answer"
