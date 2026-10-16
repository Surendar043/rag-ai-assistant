"""LangGraph graph nodes."""

from typing import List, Dict
from src.agent.state import AgentState
from src.pipeline.retrieval import Retriever
from src.pipeline.generation import Generator


def route_node(state: AgentState) -> AgentState:
    """Decide whether retrieval is needed."""
    state["needs_retrieval"] = True
    state["attempt"] = state.get("attempt", 0) + 1
    return state


def retrieve_node(state: AgentState, retriever: Retriever) -> AgentState:
    """Call the retrieve_documents tool."""
    docs = retriever.retrieve(state["question"])
    state["retrieved_docs"] = docs
    state["tool_calls"] = state.get("tool_calls", []) + ["retrieve_documents"]
    return state


def generate_node(state: AgentState, generator: Generator) -> AgentState:
    """Generate the final answer from retrieved docs."""
    answer = generator.generate(state["question"], state.get("retrieved_docs", []))
    state["final_answer"] = answer
    return state


def grade_node(state: AgentState) -> AgentState:
    """Simple grounding check on the generated answer."""
    answer = state.get("final_answer", "").lower()
    docs_text = " ".join(d["text"] for d in state.get("retrieved_docs", [])).lower()
    # Consider grounded if answer contains words from retrieved docs, or if no docs were retrieved
    if not state.get("retrieved_docs"):
        state["grounded"] = True
    else:
        answer_words = answer.split()[:5]
        state["grounded"] = bool(answer) and any(w in docs_text for w in answer_words if len(w) > 3)
    return state
