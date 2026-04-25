"""Evaluation metrics: retrieval (precision@k, recall@k, MRR, NDCG) and answer quality."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
Compare this system answer against the reference answer for a Stripe API question.
Score three things from 0.0 to 1.0:
- groundedness: does the system answer stick to facts without inventing anything?
- correctness: does it match the key facts in the reference?
- completeness: does it cover everything the reference covers?

Return JSON: {"groundedness": <float>, "correctness": <float>, "completeness": <float>}
"""


# --- Retrieval metrics ---

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    def dcg(ids: list[str]) -> float:
        score = 0.0
        for rank, doc_id in enumerate(ids[:k], start=1):
            rel = 1.0 if doc_id in relevant_ids else 0.0
            score += rel / math.log2(rank + 1)
        return score

    ideal_ids = list(relevant_ids) + [""] * k
    ideal_dcg = dcg(ideal_ids[:k])
    actual_dcg = dcg(retrieved_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    k_values = k_values or [1, 3, 5]
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4)
    metrics["mrr"] = round(mrr(retrieved_ids, relevant_ids), 4)
    return metrics


# --- Answer quality metrics ---

def llm_judge_answer(
    query: str,
    reference_answer: str,
    system_answer: str,
) -> dict[str, float]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = (
        f"Question: {query}\n"
        f"Reference answer: {reference_answer}\n"
        f"System answer: {system_answer}"
    )
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(msg.content[0].text.strip())
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return {"groundedness": 0.0, "correctness": 0.0, "completeness": 0.0}


def citation_correctness(citations: list[str], accepted_doc_ids: set[str]) -> float:
    if not citations:
        return 0.0
    cited_ids = set()
    for c in citations:
        parts = c.split("]")
        if parts:
            chunk_id = parts[0].lstrip("[")
            cited_ids.add(chunk_id)
    valid = cited_ids & accepted_doc_ids
    return len(valid) / len(cited_ids) if cited_ids else 0.0


def abstention_accuracy(abstained: bool, has_evidence: bool) -> float:
    """1.0 if abstention decision is correct (abstained when no evidence, answered when evidence exists)."""
    if abstained and not has_evidence:
        return 1.0
    if not abstained and has_evidence:
        return 1.0
    return 0.0
