"""Recursive, token-aware text chunker.

Splits text into chunks of at most `chunk_size` tokens, preferring coarse
boundaries (paragraphs -> lines -> sentences -> words) and only descending
a level when a piece is still too large. Consecutive chunks share up to
`overlap` tokens of trailing context.
"""

import tiktoken

from core.models import Chunk

_ENC = tiktoken.get_encoding("cl100k_base")

SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_keeping_sep(text: str, sep: str) -> list[str]:
    """Split on sep, keeping it attached to the preceding part, so that
    concatenating the parts reproduces the original text."""
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _split_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces of at most chunk_size tokens each, using the
    coarsest separator that gets every piece under the limit."""
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        # Nothing left to split on: hard-split by tokens.
        tokens = _ENC.encode(text)
        return [
            _ENC.decode(tokens[i : i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]
    sep, rest = separators[0], separators[1:]
    pieces: list[str] = []
    for part in _split_keeping_sep(text, sep):
        if not part.strip():
            continue
        if count_tokens(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_pieces(part, chunk_size, rest))
    return pieces


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[Chunk]:
    """Chunk text into <= chunk_size-token chunks with ~overlap-token overlap.

    Overlap is piece-aligned: the trailing pieces of a finished chunk (up to
    `overlap` tokens' worth) seed the next chunk, so no chunk exceeds
    chunk_size tokens.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pieces = _split_pieces(text, chunk_size, SEPARATORS)

    raw_chunks: list[str] = []
    window: list[str] = []  # pieces of the chunk being built
    window_tokens = 0
    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if window and window_tokens + piece_tokens > chunk_size:
            raw_chunks.append("".join(window))
            # Keep at most `overlap` tokens of tail pieces as shared context.
            while window and window_tokens > overlap:
                window_tokens -= count_tokens(window[0])
                window.pop(0)
        window.append(piece)
        window_tokens += piece_tokens
    if window:
        raw_chunks.append("".join(window))

    return [
        Chunk(content=c.strip(), token_count=count_tokens(c.strip()), chunk_index=i)
        for i, c in enumerate(raw_chunks)
    ]
