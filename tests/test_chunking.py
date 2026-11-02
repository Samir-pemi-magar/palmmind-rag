import pytest

from app.services.chunking import (
    chunk_text,
    fixed_chunking,
    recursive_chunking,
)


def test_fixed_chunking_respects_chunk_size():
    text = "A" * 250

    chunks = fixed_chunking(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 3
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_fixed_chunking_preserves_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = fixed_chunking(text, chunk_size=10, chunk_overlap=3)

    assert chunks[0][-3:] == chunks[1][:3]


def test_recursive_chunking_respects_maximum_size():
    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph explains another important concept.\n\n"
        "Third paragraph provides additional details."
    )

    chunks = recursive_chunking(
        text,
        chunk_size=60,
        chunk_overlap=10,
    )

    assert chunks
    assert all(len(chunk) <= 60 for chunk in chunks)


def test_recursive_chunking_preserves_text_content():
    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph explains another important concept."
    )

    chunks = recursive_chunking(
        text,
        chunk_size=60,
        chunk_overlap=10,
    )

    combined_text = " ".join(chunks)

    assert "First paragraph" in combined_text
    assert "Second paragraph" in combined_text


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_chunking_returns_empty_list_for_empty_text(strategy):
    assert chunk_text("   ", strategy) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunking_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        fixed_chunking("Some text", chunk_size, 0)

    with pytest.raises(ValueError):
        recursive_chunking("Some text", chunk_size, 0)


@pytest.mark.parametrize("overlap", [-1, 100])
def test_chunking_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        fixed_chunking("Some text", 100, overlap)

    with pytest.raises(ValueError):
        recursive_chunking("Some text", 100, overlap)


def test_chunk_text_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_text("Some text", "unknown")
