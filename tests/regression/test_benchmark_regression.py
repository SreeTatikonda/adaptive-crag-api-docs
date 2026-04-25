"""Regression tests: assert key metrics don't drop below baseline thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    citation_correctness,
    abstention_accuracy,
)


# --- Deterministic metric unit tests (no LLM calls) ---

def test_precision_at_k_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_zero():
    assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_precision_at_k_partial():
    result = precision_at_k(["a", "x", "b"], {"a", "b"}, k=3)
    assert abs(result - 2 / 3) < 1e-6


def test_recall_at_k_full():
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_partial():
    result = recall_at_k(["a", "x", "x"], {"a", "b"}, k=3)
    assert result == 0.5


def test_mrr_first_hit():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_second_hit():
    assert abs(mrr(["x", "a", "b"], {"a"}) - 0.5) < 1e-6


def test_mrr_no_hit():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect():
    result = ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
    assert abs(result - 1.0) < 1e-6


def test_ndcg_partial():
    result = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert 0.0 < result < 1.0


def test_citation_correctness_all_valid():
    citations = ["[abc12345678901234] https://stripe.com"]
    accepted_ids = {"abc12345678901234"}
    assert citation_correctness(citations, accepted_ids) == 1.0


def test_citation_correctness_none_valid():
    citations = ["[zzz] https://stripe.com"]
    accepted_ids = {"abc12345678901234"}
    assert citation_correctness(citations, accepted_ids) == 0.0


def test_citation_correctness_empty():
    assert citation_correctness([], {"abc"}) == 0.0


def test_abstention_accuracy_correct_abstain():
    assert abstention_accuracy(abstained=True, has_evidence=False) == 1.0


def test_abstention_accuracy_correct_answer():
    assert abstention_accuracy(abstained=False, has_evidence=True) == 1.0


def test_abstention_accuracy_wrong_abstain():
    assert abstention_accuracy(abstained=True, has_evidence=True) == 0.0


def test_abstention_accuracy_wrong_answer():
    assert abstention_accuracy(abstained=False, has_evidence=False) == 0.0


# --- Regression threshold test (uses saved results if they exist) ---

RESULTS_DIR = Path("data/processed/eval_results")


@pytest.mark.skipif(
    not (RESULTS_DIR / "adaptive_crag_per_question.json").exists(),
    reason="No benchmark results found; run `make benchmark` first",
)
def test_adaptive_crag_meets_baseline_thresholds():
    results_path = RESULTS_DIR / "adaptive_crag_per_question.json"
    per_q = json.loads(results_path.read_text())
    valid = [r for r in per_q if "error" not in r]

    if not valid:
        pytest.skip("No valid results to evaluate")

    avg_groundedness = sum(r.get("groundedness", 0) for r in valid) / len(valid)
    avg_abstention_acc = sum(r.get("abstention_accuracy", 0) for r in valid) / len(valid)

    assert avg_groundedness >= 0.5, f"Groundedness {avg_groundedness:.2f} below threshold 0.5"
    assert avg_abstention_acc >= 0.5, f"Abstention accuracy {avg_abstention_acc:.2f} below threshold 0.5"
