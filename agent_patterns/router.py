"""LangGraph query-routing agent: classify the question, then route to a
rag / direct / reject handler."""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

_ROUTES = {"rag", "direct", "reject"}

_CLASSIFY = ChatPromptTemplate.from_template(
    "Classify the user question into exactly one label: rag, direct, or reject.\n"
    "- rag: needs looking up documents or knowledge\n"
    "- direct: a general question answerable directly\n"
    "- reject: harmful, nonsensical, or out of scope\n"
    "Answer with ONLY the label.\n\nQuestion: {question}"
)
_ANSWER = ChatPromptTemplate.from_template(
    "Answer the question.\n\nQuestion: {question}\nAnswer:"
)


class RouterState(TypedDict):
    question: str
    route: str
    answer: str


def build_router_graph(llm):
    classify_chain = _CLASSIFY | llm | StrOutputParser()
    answer_chain = _ANSWER | llm | StrOutputParser()

    def classify(state: RouterState) -> dict:
        label = classify_chain.invoke({"question": state["question"]}).strip().lower()
        return {"route": label if label in _ROUTES else "direct"}

    def rag_node(state: RouterState) -> dict:
        # In practice this calls the RAG pipeline; here the LLM answers as the rag path.
        return {"answer": "[rag] " + answer_chain.invoke({"question": state["question"]})}

    def direct_node(state: RouterState) -> dict:
        return {"answer": "[direct] " + answer_chain.invoke({"question": state["question"]})}

    def reject_node(state: RouterState) -> dict:
        return {"answer": "I can't help with that request."}

    g = StateGraph(RouterState)
    g.add_node("classify", classify)
    g.add_node("rag", rag_node)
    g.add_node("direct", direct_node)
    g.add_node("reject", reject_node)
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", lambda s: s["route"],
        {"rag": "rag", "direct": "direct", "reject": "reject"},
    )
    g.add_edge("rag", END)
    g.add_edge("direct", END)
    g.add_edge("reject", END)
    return g.compile()
