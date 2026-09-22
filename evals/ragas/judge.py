"""Judge model factory for Ragas. Local Ollama by default; API override via
RAGAS_JUDGE=openai|anthropic. Returns Ragas-wrapped (llm, embeddings)."""

import os

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


def build_judge():
    provider = os.getenv("RAGAS_JUDGE", "ollama").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        emb = OpenAIEmbeddings(model=os.getenv("RAGAS_OPENAI_EMBED", "text-embedding-3-small"))
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from langchain_ollama import OllamaEmbeddings

        llm = ChatAnthropic(model=os.getenv("RAGAS_ANTHROPIC_MODEL", "claude-sonnet-4-5"), temperature=0)
        emb = OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:  # local Ollama (default, $0)
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=os.getenv("LLM_MODEL", "llama3.2"), base_url=base, temperature=0)
        emb = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"), base_url=base)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)
