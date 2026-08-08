from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from langgraph_rag.graph import build_graph


def test_graph_compiles_with_expected_nodes():
    dummy_retriever = RunnableLambda(lambda q: [])
    llm = FakeListChatModel(responses=["yes"])
    graph = build_graph(dummy_retriever, llm)
    node_names = set(graph.get_graph().nodes)
    assert {"retrieve", "grade_documents", "generate", "transform_query"} <= node_names
