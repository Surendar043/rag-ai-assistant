"""Evaluation metrics.

Retrieval quality (recall@k, MRR) is measured separately from answer quality
(correctness, grounding), matching the project's evaluation goal.
"""

from typing import List, Dict


def recall_at_k(retrieved: List[Dict], expected_sources: List[str], k: int) -> float:
    """Fraction of expected sources found in the top-k retrieved chunks."""
    if not expected_sources:
        return 1.0 if not retrieved else 0.0
    retrieved_sources = {d.get("source") for d in retrieved[:k]}
    hits = len(retrieved_sources & set(expected_sources))
    return hits / len(expected_sources)


def mrr(retrieved: List[Dict], expected_sources: List[str]) -> float:
    """Mean Reciprocal Rank of the first relevant source."""
    if not expected_sources:
        return 1.0 if not retrieved else 0.0
    expected_set = set(expected_sources)
    for rank, d in enumerate(retrieved, start=1):
        if d.get("source") in expected_set:
            return 1.0 / rank
    return 0.0


def answer_correctness(generated: str, expected: str) -> float:
    """Simple correctness check: exact match or expected contained in generated."""
    gen = generated.lower().strip()
    exp = expected.lower().strip()
    if gen == exp or exp in gen:
        return 1.0
    # Partial credit for key overlap
    gen_words = set(gen.split())
    exp_words = set(exp.split())
    if exp_words and gen_words:
        return len(gen_words & exp_words) / len(exp_words)
    return 0.0


def is_grounded(generated: str, retrieved: List[Dict]) -> bool:
    """Heuristic grounding check: does the answer reference retrieved content?"""
    gen = generated.lower()
    docs_text = " ".join(d.get("text", "") for d in retrieved).lower()
    if not docs_text:
        return "don't know" in gen or "i don't know" in gen
    gen_words = [w for w in gen.split() if len(w) > 3]
    return any(w in docs_text for w in gen_words)
