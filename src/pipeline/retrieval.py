"""Retrieval layer: query the vector store for relevant chunks."""

from typing import List, Dict, Optional
from src.pipeline.vector_store import VectorStore
from src.config.settings import settings


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Return top-k relevant chunks with scores."""
        return self.vs.search(query, top_k=top_k)
