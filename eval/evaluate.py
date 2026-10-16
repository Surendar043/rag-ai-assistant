"""Run the evaluation suite.

Measures retrieval quality (recall@k, MRR) separately from answer quality
(correctness, grounding) across the structured dataset and edge cases.
"""

import json
import sys
from pathlib import Path

# Allow running as a script from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import recall_at_k, mrr, answer_correctness, is_grounded
from src.agent.graph import run_agent
from src.pipeline.retrieval import Retriever
from src.pipeline.vector_store import VectorStore
from src.models.embedding_model import EmbeddingModel
from src.models.llm import LLM
from src.pipeline.generation import Generator
from src.config.settings import settings


def load_components():
    emb = EmbeddingModel(settings.EMBEDDING_MODEL)
    vs = VectorStore(emb)
    try:
        vs.load(settings.VECTOR_STORE_PATH)
    except Exception as exc:
        print(f"WARNING: Could not load vector store: {exc}")
        print("Index documents first via POST /ingest, then re-run evaluation.")
        sys.exit(1)
    retriever = Retriever(vs)
    llm = LLM(settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
    generator = Generator(llm)
    return retriever, generator


def evaluate_dataset(retriever, generator):
    with open(settings.EVAL_DATASET) as f:
        dataset = json.load(f)

    retrieval_scores = []
    answer_scores = []
    grounding_scores = []

    print("\n=== Structured Dataset Evaluation ===")
    for item in dataset:
        question = item["question"]
        docs = retriever.retrieve(question, top_k=settings.TOP_K)
        answer = run_agent(question, retriever, generator)

        r_score = recall_at_k(docs, item.get("expected_sources", []), settings.TOP_K)
        m_score = mrr(docs, item.get("expected_sources", []))
        a_score = answer_correctness(answer, item["expected_answer"])
        g_score = is_grounded(answer, docs)

        retrieval_scores.append(r_score)
        answer_scores.append(a_score)
        grounding_scores.append(g_score)

        print(
            f"[{item['id']}] {question[:50]}... | "
            f"recall@{settings.TOP_K}={r_score:.2f} MRR={m_score:.2f} "
            f"correctness={a_score:.2f} grounded={g_score}"
        )

    n = len(dataset)
    print("\n--- Aggregate ---")
    print(f"Retrieval Recall@{settings.TOP_K}: {sum(retrieval_scores)/n:.3f}")
    print(f"Retrieval MRR: {sum(mrr(docs, item.get('expected_sources', [])) for docs, item in zip([retriever.retrieve(i['question']) for i in dataset], dataset))/n:.3f}")
    print(f"Answer Correctness: {sum(answer_scores)/n:.3f}")
    print(f"Answer Grounded: {sum(grounding_scores)/n:.3f}")


def evaluate_edge_cases(retriever, generator):
    with open(settings.EVAL_EDGE_CASES) as f:
        edge_cases = json.load(f)

    print("\n=== Edge Case Evaluation ===")
    for item in edge_cases:
        question = item["question"]
        expected = item["expected_behavior"]
        try:
            answer = run_agent(question, retriever, generator)
        except Exception as exc:
            answer = f"ERROR: {exc}"
        print(f"[{item['id']}] expected={expected} | answer={answer[:80]}...")


def main():
    retriever, generator = load_components()
    evaluate_dataset(retriever, generator)
    evaluate_edge_cases(retriever, generator)


if __name__ == "__main__":
    main()
