"""Entrypoint for LangGraph Studio (`langgraph dev`). Builds the real agent
(pgvector + Ollama) so the graph can be visualized and run interactively.
Requires Postgres + Ollama up (docker compose up -d)."""

from core.db import VectorStore
from core.embedder import Embedder
from langgraph_rag.graph import build_agent


def make_graph():
    # Studio calls this once to obtain the compiled graph. The VectorStore
    # connection lives for the dev-server session.
    return build_agent(VectorStore(), Embedder())
