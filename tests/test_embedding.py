"""Tests for the embedding model (offline fallback safe)."""

from src.models.embedding_model import EmbeddingModel


def test_embedding_shape():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    vectors = emb.embed(["hello world", "foo bar"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == emb.dim
    assert emb.dim > 0


def test_embedding_deterministic():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    v1 = emb.embed(["same text"])
    v2 = emb.embed(["same text"])
    assert (v1 == v2).all()


def test_embedding_different_texts():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    v1 = emb.embed(["apple banana"])
    v2 = emb.embed(["cherry date"])
    assert not (v1 == v2).all()
