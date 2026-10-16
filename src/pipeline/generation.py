"""Response generation: build prompt and call the LLM."""

from typing import List, Dict
from src.models.llm import LLM
from src.config.settings import settings


def build_prompt(query: str, contexts: List[Dict]) -> str:
    """Build a grounded prompt from the query and retrieved contexts."""
    ctx_parts = []
    for i, c in enumerate(contexts):
        ctx_parts.append(f"[Source {i+1}] {c.get('text', '')}")
    ctx = "\n\n".join(ctx_parts)
    return (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


class Generator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """Generate an answer grounded in the retrieved contexts."""
        prompt = build_prompt(query, contexts)
        return self.llm.complete(prompt)
