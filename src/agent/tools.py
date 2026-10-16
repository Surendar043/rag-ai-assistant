"""Agent tools: callable functions the LangGraph agent can invoke."""

from typing import List, Dict
from src.pipeline.retrieval import Retriever


def retrieve_documents(retriever: Retriever, question: str) -> List[Dict]:
    """Tool: retrieve relevant document chunks for a question."""
    return retriever.retrieve(question)
