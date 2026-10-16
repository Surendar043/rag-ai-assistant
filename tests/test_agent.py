"""Tests for the LangGraph agent."""

from src.pipeline.retrieval import Retriever
from src.pipeline.vector_store import VectorStore
from src.pipeline.generation import Generator
from src.models.embedding_model import EmbeddingModel
from src.models.llm import LLM
from src.agent.graph import run_agent


def test_agent_returns_answer():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    vs = VectorStore(emb)
    vs.add_chunks(
        [{"text": "The refund policy is 30 days with a receipt.", "source": "policy.pdf"}]
    )
    retriever = Retriever(vs)
    llm = LLM("mock-model", temperature=0.0)
    generator = Generator(llm)

    answer = run_agent("What is the refund policy?", retriever, generator)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_agent_no_docs():
    emb = EmbeddingModel("all-MiniLM-L6-v2")
    vs = VectorStore(emb)
    retriever = Retriever(vs)
    llm = LLM("mock-model", temperature=0.0)
    generator = Generator(llm)

    answer = run_agent("Something not in docs", retriever, generator)
    assert isinstance(answer, str)
