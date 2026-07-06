import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "vectordb"),
        user=os.getenv("POSTGRES_USER", "vectordb"),
        password=os.getenv("POSTGRES_PASSWORD", "vectordb"),
    )
    register_vector(conn)
    return conn


class VectorStore:
    def __init__(self):
        self.conn = get_connection()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # --- Documents ---

    def insert_document(self, content: str, source: str | None, embedding: list[float]) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s) RETURNING id",
                (content, source, embedding),
            )
            row_id = cur.fetchone()[0]
        self.conn.commit()
        return row_id

    def search_documents(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, content, source,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            return [dict(r) for r in cur.fetchall()]

    def clear_documents(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents")
        self.conn.commit()

    # --- Products ---

    def insert_product(
        self,
        name: str,
        description: str,
        category: str | None,
        price: float | None,
        embedding: list[float],
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO products (name, description, category, price, embedding)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (name, description, category, price, embedding),
            )
            row_id = cur.fetchone()[0]
        self.conn.commit()
        return row_id

    def search_products(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, category, price,
                       1 - (embedding <=> %s::vector) AS score
                FROM products
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            return [dict(r) for r in cur.fetchall()]

    def clear_products(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM products")
        self.conn.commit()

    # --- Logs ---

    def insert_log(
        self,
        message: str,
        level: str | None,
        service: str | None,
        embedding: list[float],
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO logs (message, level, service, embedding)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (message, level, service, embedding),
            )
            row_id = cur.fetchone()[0]
        self.conn.commit()
        return row_id

    def search_logs(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, message, level, service,
                       1 - (embedding <=> %s::vector) AS score
                FROM logs
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_all_logs_with_embeddings(self) -> list[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, message, level, service, embedding FROM logs")
            return [dict(r) for r in cur.fetchall()]

    def clear_logs(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM logs")
        self.conn.commit()
