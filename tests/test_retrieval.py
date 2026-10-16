"""Tests for retrieval."""

from src.pipeline.retrieval import Retriever
from src.pipeline.vector_store import VectorStore
from src.models.embedding_model import EmbeddingModel


def test_retrieval_returns_relevant_chunk():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    vs = VectorStore(emb)
    vs.add_chunks(
        [{"text": "The refund policy is 30 days.", "source": "policy.pdf"}]
    )
    retriever = Retriever(vs)
    results = retriever.retrieve("refund policy days")
    assert len(results) > 0
    assert "refund" in results[0]["text"].lower() or "policy" in results[0]["text"].lower()


def test_retrieval_empty_index():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    vs = VectorStore(emb)
    retriever = Retriever(vs)
    results = retriever.retrieve("anything")
    assert results == []


def test_retrieval_has_scores():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    vs = VectorStore(emb)
    vs.add_chunks([{"text": "Sample document text.", "source": "s.pdf"}])
    retriever = Retriever(vs)
    results = retriever.retrieve("sample")
    assert len(results) > 0
    assert "score" in results[0]
    assert 0.0 <= results[0]["score"] <= 1.0
