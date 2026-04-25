from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from src.graph.state import GradeResult, RAGState

logger = logging.getLogger(__name__)

_GRADE_PROMPT = """\
Read this Stripe API documentation chunk and score how well it helps answer the developer's question.

Score each dimension from 0.0 to 1.0:
- relevance: does this chunk actually address what they're asking?
- sufficiency: does it have enough detail to answer on its own, or is it just tangential?
- specificity: is it specific to the exact endpoint/behavior/parameter asked about, not generic boilerplate?
- version_match: if the question names a version, does this chunk match? (give 1.0 if no version was specified)

Decide "accept" if this chunk is worth using in the answer, "reject" if not.
Accept when relevance >= 0.5 and (sufficiency + specificity) >= 0.8.

Return JSON:
{
  "relevance": <float>,
  "sufficiency": <float>,
  "specificity": <float>,
  "version_match": <float>,
  "decision": "accept" or "reject",
  "rationale": "<brief reason>"
}
"""


def _grade_single(
    query: str,
    doc: dict[str, Any],
    client: anthropic.Anthropic,
    constraints: dict[str, Any],
) -> GradeResult:
    chunk_text = doc["text"][:2000]
    section = " > ".join(doc.get("section_path", []))
    version_info = f"Chunk version: {doc.get('version', 'unspecified')}"
    query_version = constraints.get("api_version", "not specified")

    prompt = (
        f"Developer question: {query}\n"
        f"Version they asked about: {query_version}\n"
        f"Doc section: {section}\n"
        f"{version_info}\n\n"
        f"Chunk text:\n{chunk_text}"
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0,
        system=_GRADE_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        return GradeResult(
            doc_id=doc["doc_id"],
            chunk_id=doc["chunk_id"],
            relevance=float(parsed.get("relevance", 0.0)),
            sufficiency=float(parsed.get("sufficiency", 0.0)),
            specificity=float(parsed.get("specificity", 0.0)),
            version_match=float(parsed.get("version_match", 1.0)),
            decision=parsed.get("decision", "reject"),
            rationale=parsed.get("rationale", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return GradeResult(
            doc_id=doc["doc_id"],
            chunk_id=doc["chunk_id"],
            relevance=0.0,
            sufficiency=0.0,
            specificity=0.0,
            version_match=1.0,
            decision="reject",
            rationale="Failed to parse grader response",
        )


def grade_documents(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    retrieved = state.get("retrieved_docs", [])
    constraints = state.get("constraints", {})
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    if not retrieved:
        trace.append("grade_documents → no docs to grade")
        return {
            "grade_results": [],
            "accepted_docs": [],
            "rejected_docs": [],
            "trace": trace,
            "metrics": metrics,
        }

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    grade_results: list[dict] = []
    accepted: list[dict] = []
    rejected: list[dict] = []

    for doc in retrieved[:8]:
        result = _grade_single(query, doc, client, constraints)
        grade_results.append(result.model_dump())

        if result.decision == "accept":
            accepted.append(doc)
        else:
            rejected.append(doc)

        metrics["input_tokens"] = metrics.get("input_tokens", 0)
        metrics["output_tokens"] = metrics.get("output_tokens", 0)

    avg_relevance = sum(g["relevance"] for g in grade_results) / len(grade_results) if grade_results else 0.0

    trace.append(
        f"grade_documents → {len(accepted)} accepted, {len(rejected)} rejected "
        f"(avg_relevance={avg_relevance:.2f})"
    )
    logger.info("Graded %d docs: %d accepted, %d rejected", len(retrieved[:8]), len(accepted), len(rejected))

    return {
        "grade_results": grade_results,
        "accepted_docs": accepted,
        "rejected_docs": rejected,
        "trace": trace,
        "metrics": metrics,
    }
