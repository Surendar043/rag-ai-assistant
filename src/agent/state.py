"""Agent state shared across LangGraph nodes."""

from typing import List, Dict, Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    retrieved_docs: List[Dict]
    tool_calls: List[str]
    final_answer: str
    needs_retrieval: bool
    grounded: bool
    attempt: int
