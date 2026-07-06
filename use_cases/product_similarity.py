"""
Use case: Product / Item Similarity
-------------------------------------
Given a product description, find the most semantically similar products
in the catalog. Useful for "customers also viewed" or search ranking.
"""

from core.db import VectorStore
from core.embedder import Embedder
from core.models import Product, ProductResult


def _product_text(p: Product) -> str:
    return f"{p.name}. {p.description}"


def index_products(products: list[Product], store: VectorStore, embedder: Embedder) -> None:
    store.clear_products()
    for p in products:
        embedding = embedder.embed(_product_text(p))
        store.insert_product(p.name, p.description, p.category, p.price, embedding)
    print(f"Indexed {len(products)} products.")


def find_similar(query: str, store: VectorStore, embedder: Embedder, top_k: int = 5) -> list[ProductResult]:
    embedding = embedder.embed(query)
    rows = store.search_products(embedding, top_k=top_k)
    return [ProductResult(**r) for r in rows]


def run(store: VectorStore, embedder: Embedder) -> None:
    from data.products import SAMPLE_PRODUCTS

    print("\n=== Product Similarity ===")
    index_products(SAMPLE_PRODUCTS, store, embedder)

    queries = [
        "lightweight running shoes for marathon training",
        "noise-cancelling headphones for travel",
        "ergonomic chair for home office",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = find_similar(query, store, embedder)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.name} (${r.price}) — {r.category}")
