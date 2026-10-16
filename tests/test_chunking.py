"""Tests for document chunking."""

from src.pipeline.chunking import chunk_text, chunk_documents


def test_chunk_text_basic():
    text = "word " * 600
    chunks = chunk_text(text, source="test.txt")
    assert len(chunks) > 1
    assert all(c["source"] == "test.txt" for c in chunks)
    assert all("text" in c and "chunk_id" in c for c in chunks)


def test_chunk_text_overlap():
    text = "word " * 600
    chunks = chunk_text(text, source="s.txt")
    # With CHUNK_SIZE=500 and CHUNK_OVERLAP=50, expect at least 2 chunks
    assert len(chunks) >= 2


def test_chunk_documents():
    docs = [
        {"text": "Hello world this is a test document.", "source": "a.txt"},
        {"text": "Another document with different content here.", "source": "b.txt"},
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 2
    sources = {c["source"] for c in chunks}
    assert sources == {"a.txt", "b.txt"}


def test_chunk_empty_text():
    chunks = chunk_text("", source="empty.txt")
    assert chunks == []
