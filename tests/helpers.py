"""Shared test utilities: a stdlib-only minimal PDF writer and a
deterministic fake embedder (no Ollama needed)."""

import hashlib
import struct


def _purge_test_rows(store) -> None:
    """Delete all sources whose path starts with 'zz-test', regardless of root."""
    with store.conn.cursor() as cur:
        cur.execute("DELETE FROM sources WHERE path LIKE 'zz-test%'")
    store.conn.commit()


def fake_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic pseudo-embedding: same text -> same vector. Lets DB and
    ingestion tests run without Ollama."""
    digest = hashlib.sha256(text.encode()).digest()
    data = digest * (dim * 4 // len(digest) + 1)
    return [
        struct.unpack_from("<I", data, i * 4)[0] % 1000 / 1000.0 for i in range(dim)
    ]


class FakeEmbedder:
    """Drop-in for core.embedder.Embedder in tests."""

    def embed(self, text: str) -> list[float]:
        return fake_embedding(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def build_pdf(text: str) -> bytes:
    """Build a tiny one-page PDF containing `text` (Helvetica, no escaping —
    keep text free of parentheses and backslashes)."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_pos,
    )
    return bytes(out)
