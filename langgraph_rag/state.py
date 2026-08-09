"""Agent state shared across LangGraph nodes."""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    attempts: int
