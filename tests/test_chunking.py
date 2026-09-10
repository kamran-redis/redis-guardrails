import pytest

from redis_guardrails.chunking import chunk_text
from redis_guardrails.errors import IncompleteCoverageError


def test_short_text_returns_single_chunk():
    chunks = chunk_text("hello world", source="input")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].evaluated_text == "hello world"
    assert chunks[0].start_character == 0
    assert chunks[0].end_character == len("hello world")
    assert chunks[0].id == "input-0"


def test_prefix_is_included_in_evaluated_text_but_not_text():
    chunks = chunk_text(
        "hello world", source="output", prefix="User: hi\nAssistant: "
    )
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].evaluated_text == "User: hi\nAssistant: hello world"


def test_prefix_that_exceeds_max_chars_raises():
    with pytest.raises(IncompleteCoverageError):
        chunk_text("hi", source="output", prefix="x" * 100, max_chars=50)


def test_long_text_splits_with_full_coverage_and_overlap():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 40  # long enough to require multiple chunks
    chunks = chunk_text(text, source="input", max_chars=200, overlap_chars=20)

    assert len(chunks) > 1
    # Full coverage: every character index is covered by at least one chunk.
    covered = [False] * len(text)
    for c in chunks:
        for i in range(c.start_character, c.end_character):
            covered[i] = True
    assert all(covered)

    # Overlap: consecutive chunks share some text.
    for first, second in zip(chunks, chunks[1:]):
        assert second.start_character < first.end_character


def test_prefers_paragraph_boundary_over_max_chars_cutoff():
    text = "A" * 90 + "\n\n" + "B" * 90
    chunks = chunk_text(text, source="input", max_chars=100, overlap_chars=5)
    # The first chunk should end exactly at the paragraph boundary (index 92),
    # not at the raw max_chars cutoff (index 100), since 92 <= 100.
    assert chunks[0].end_character == 92


def test_single_long_word_with_no_boundary_raises():
    with pytest.raises(IncompleteCoverageError):
        chunk_text("supercalifragilisticexpialidocious", source="input", max_chars=10)


def test_exceeding_max_chunks_raises():
    text = "word " * 1000
    with pytest.raises(IncompleteCoverageError):
        chunk_text(text, source="input", max_chars=20, overlap_chars=2, max_chunks=3)
