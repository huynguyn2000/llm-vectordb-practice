"""Assemble the CRAG-lite agent graph."""

import os

from langgraph.graph import END, START, StateGraph

from langgraph_rag.nodes import make_decider, make_nodes
from langgraph_rag.state import AgentState


def build_graph(retriever, llm, max_attempts: int = 2):
    nodes = make_nodes(retriever, llm, max_attempts=max_attempts)
    decide = make_decider(max_attempts=max_attempts)

    g = StateGraph(AgentState)
    g.add_node("retrieve", nodes["retrieve"])
    g.add_node("grade_documents", nodes["grade_documents"])
    g.add_node("generate", nodes["generate"])
    g.add_node("transform_query", nodes["transform_query"])

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    g.add_edge("transform_query", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def build_agent(store, embedder, top_k: int = 3, max_attempts: int = 2):
    # Imported here so the module loads without langchain_rag/ollama at import time.
    from langchain_ollama import ChatOllama

    from langchain_rag.retriever import HybridRetriever

    retriever = HybridRetriever(store=store, embedder=embedder, top_k=top_k)
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    return build_graph(retriever, llm, max_attempts=max_attempts)
