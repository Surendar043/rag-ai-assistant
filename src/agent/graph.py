"""LangGraph StateGraph: routes, retrieves, generates, grades, retries."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import route_node, retrieve_node, generate_node, grade_node
from src.pipeline.retrieval import Retriever
from src.pipeline.generation import Generator


def build_graph(retriever: Retriever, generator: Generator):
    workflow = StateGraph(AgentState)

    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    workflow.add_node("generate", lambda s: generate_node(s, generator))
    workflow.add_node("grade", grade_node)

    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")

    def should_retry(state: AgentState):
        if state.get("attempt", 0) >= 2:
            return END
        if not state.get("grounded") and not state.get("retrieved_docs"):
            return "retrieve"
        return END

    workflow.add_conditional_edges("grade", should_retry)
    return workflow.compile()


def run_agent(question: str, retriever: Retriever, generator: Generator) -> str:
    """Run the agent on a single question and return the final answer."""
    graph = build_graph(retriever, generator)
    state: AgentState = {"question": question, "attempt": 0}
    result = graph.invoke(state)
    return result.get("final_answer", "I couldn't find an answer.")
