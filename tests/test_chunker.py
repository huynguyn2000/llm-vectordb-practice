from ingestion.chunker import chunk_text, count_tokens


def test_count_tokens_positive():
    assert 0 < count_tokens("hello world") <= len("hello world")


def test_empty_and_whitespace_input_yield_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_stripped_chunk():
    # Chunk content is stripped of surrounding whitespace (e.g. a trailing newline).
    text = "Para one.\n\nPara two.\n"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text.strip()
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count == count_tokens(text.strip())


def test_chunks_never_exceed_chunk_size():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_chunk_indexes_are_sequential():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_splits_fall_on_sentence_boundaries():
    # One giant paragraph (no \n\n) forces descent to the ". " level.
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(c.content.endswith(".") for c in chunks)


def test_consecutive_chunks_share_overlap():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        first_sentence_of_b = b.content.split(".")[0] + "."
        assert first_sentence_of_b in a.content


def test_zero_overlap_means_disjoint_chunks():
    text = " ".join(f"Sentence number {i:03d} ends now." for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    for a, b in zip(chunks, chunks[1:]):
        first_sentence_of_b = b.content.split(".")[0] + "."
        assert first_sentence_of_b not in a.content


def test_unsplittable_text_hard_splits_by_tokens():
    text = "x" * 5000  # no separators at all
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_oversized_piece_after_overlap_carry_stays_within_limit():
    # Small sentences leave a short overlap tail; the following hard-split
    # piece nearly fills chunk_size on its own. Tail + piece must not
    # produce an oversized chunk.
    text = "a. b. " + "x" * 2000
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert all(c.token_count <= 100 for c in chunks)
