import os
import ollama
from dotenv import load_dotenv

load_dotenv()


class Embedder:
    def __init__(self):
        self.model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
