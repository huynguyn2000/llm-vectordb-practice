"""Mem0 configuration — fully local (Ollama LLM + embedder + Chroma). Pure: no
mem0 import, so it is unit-testable without the memory extra installed."""

import os


def build_memory_config() -> dict:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": os.getenv("LLM_MODEL", "llama3.2"),
                "ollama_base_url": base,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
                "ollama_base_url": base,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "mem0", "path": ".mem0_chroma"},
        },
    }
