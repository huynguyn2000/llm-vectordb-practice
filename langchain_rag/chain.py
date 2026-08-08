"""LangChain LCEL RAG chain over the project's hybrid retrieval, generating with
a local Ollama model. LangSmith tracing activates from env vars (off by default);
the chain carries a run name + tags so traces are legible."""

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama import ChatOllama

from core.db import VectorStore
from core.embedder import Embedder
from langchain_rag.retriever import HybridRetriever

_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['source_path']}#chunk{d.metadata['chunk_index']}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(store: VectorStore, embedder: Embedder, top_k: int = 3) -> Runnable:
    retriever = HybridRetriever(store=store, embedder=embedder, top_k=top_k)
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.with_config(run_name="langchain_rag", tags=["rag", "hybrid"])
