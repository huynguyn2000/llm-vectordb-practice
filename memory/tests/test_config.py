from memory.config import build_memory_config


def test_config_is_fully_local():
    cfg = build_memory_config()
    assert cfg["llm"]["provider"] == "ollama"
    assert cfg["embedder"]["provider"] == "ollama"
    assert cfg["vector_store"]["provider"] == "chroma"
    assert cfg["llm"]["config"]["model"]          # non-empty model
    assert cfg["embedder"]["config"]["model"]


def test_config_honors_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:1234")
    cfg = build_memory_config()
    assert cfg["llm"]["config"]["model"] == "llama3.1"
    assert cfg["llm"]["config"]["ollama_base_url"] == "http://example:1234"
