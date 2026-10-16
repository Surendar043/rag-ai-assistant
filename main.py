"""Local CLI runner for the RAG AI Assistant."""

from src.pipeline.vector_store import VectorStore
from src.pipeline.retrieval import Retriever
from src.pipeline.generation import Generator
from src.models.embedding_model import EmbeddingModel
from src.models.llm import LLM
from src.agent.graph import run_agent
from config.settings import settings


def main():
    print("=== RAG AI Assistant (CLI) ===")
    print("Type your question. Type 'exit' to quit.\n")

    emb = EmbeddingModel(settings.EMBEDDING_MODEL)
    vs = VectorStore(emb)
    try:
        vs.load(settings.VECTOR_STORE_PATH)
        print(f"Loaded existing index from {settings.VECTOR_STORE_PATH}")
    except Exception:
        print("No index found. Please index documents first via the API (/ingest).")
        return

    retriever = Retriever(vs)
    llm = LLM(settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
    generator = Generator(llm)

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
        answer = run_agent(question, retriever, generator)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
