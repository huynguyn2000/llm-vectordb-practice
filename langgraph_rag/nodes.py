"""LangGraph node factories. make_nodes closes over an injected retriever and
chat model, so production wires real ones and tests wire fakes."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph_rag.state import AgentState

_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "Is the following document relevant to the question? Answer only 'yes' or 'no'.\n\n"
    "Question: {question}\n\nDocument: {document}"
)
_GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""
)
_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the question to improve document retrieval. Return only the rewritten "
    "question.\n\nQuestion: {question}"
)


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_path', '?')}#chunk{d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
        for d in docs
    )


def make_nodes(retriever, llm, max_attempts: int = 2) -> dict:
    grade_chain = _GRADE_PROMPT | llm | StrOutputParser()
    gen_chain = _GEN_PROMPT | llm | StrOutputParser()
    rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()

    def retrieve(state: AgentState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "attempts": state.get("attempts", 0) + 1}

    def grade_documents(state: AgentState) -> dict:
        kept = []
        for d in state["documents"]:
            verdict = grade_chain.invoke(
                {"question": state["question"], "document": d.page_content}
            )
            if verdict.strip().lower().startswith("y"):
                kept.append(d)
        return {"documents": kept}

    def generate(state: AgentState) -> dict:
        context = _format_docs(state["documents"])
        text = gen_chain.invoke({"context": context, "question": state["question"]})
        return {"generation": text}

    def transform_query(state: AgentState) -> dict:
        better = rewrite_chain.invoke({"question": state["question"]})
        return {"question": better.strip()}

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "generate": generate,
        "transform_query": transform_query,
    }


def make_decider(max_attempts: int = 2):
    def decide_to_generate(state: AgentState) -> str:
        if state["documents"]:
            return "generate"
        if state.get("attempts", 0) >= max_attempts:
            return "generate"
        return "transform_query"

    return decide_to_generate
