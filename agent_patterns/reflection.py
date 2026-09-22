"""LangGraph reflection agent: generate a draft, critique it, and revise until
approved or a bounded number of revisions."""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

_GENERATE = ChatPromptTemplate.from_template(
    "Write a short response to the task.\n\nTask: {task}\nResponse:"
)
_REFLECT = ChatPromptTemplate.from_template(
    "Critique the draft for the task. If it is good enough, reply with exactly "
    "'APPROVE'. Otherwise give one concrete improvement.\n\n"
    "Task: {task}\n\nDraft: {draft}\n\nCritique:"
)
_REVISE = ChatPromptTemplate.from_template(
    "Revise the draft using the critique.\n\nTask: {task}\n\nDraft: {draft}\n\n"
    "Critique: {critique}\n\nRevised:"
)


class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    revisions: int
    approved: bool


def build_reflection_graph(llm, max_revisions: int = 2):
    gen = _GENERATE | llm | StrOutputParser()
    reflect_chain = _REFLECT | llm | StrOutputParser()
    revise_chain = _REVISE | llm | StrOutputParser()

    def generate(state: ReflectionState) -> dict:
        return {"draft": gen.invoke({"task": state["task"]}), "revisions": 0, "approved": False}

    def reflect(state: ReflectionState) -> dict:
        c = reflect_chain.invoke({"task": state["task"], "draft": state["draft"]}).strip()
        return {"critique": c, "approved": c.upper().startswith("APPROVE")}

    def revise(state: ReflectionState) -> dict:
        new = revise_chain.invoke(
            {"task": state["task"], "draft": state["draft"], "critique": state["critique"]}
        )
        return {"draft": new, "revisions": state["revisions"] + 1}

    def decide(state: ReflectionState) -> str:
        if state["approved"] or state["revisions"] >= max_revisions:
            return "end"
        return "revise"

    g = StateGraph(ReflectionState)
    g.add_node("generate", generate)
    g.add_node("reflect", reflect)
    g.add_node("revise", revise)
    g.add_edge(START, "generate")
    g.add_edge("generate", "reflect")
    g.add_conditional_edges("reflect", decide, {"revise": "revise", "end": END})
    g.add_edge("revise", "reflect")
    return g.compile()
