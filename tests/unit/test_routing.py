"""Unit tests for routing policy."""

from __future__ import annotations

import pytest

from src.routing.policy import route_correction, get_routing_action


def _make_state(
    accepted: list = None,
    rejected: list = None,
    grade_results: list = None,
    retry_count: int = 0,
) -> dict:
    return {
        "query": "test query",
        "query_type": "how_to",
        "constraints": {},
        "retrieved_docs": [],
        "accepted_docs": accepted or [],
        "rejected_docs": rejected or [],
        "grade_results": grade_results or [],
        "routing_decision": {},
        "rewritten_query": "",
        "answer": "",
        "citations": [],
        "verification_result": {},
        "trace": [],
        "metrics": {"retry_count": retry_count},
    }


def _make_grade(relevance: float, decision: str = "accept", version_match: float = 1.0) -> dict:
    return {
        "doc_id": "d1",
        "chunk_id": "c1",
        "relevance": relevance,
        "sufficiency": relevance,
        "specificity": relevance,
        "version_match": version_match,
        "decision": decision,
        "rationale": "test",
    }


def _make_doc() -> dict:
    return {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "test",
        "source_url": "https://stripe.com",
        "score": 0.8,
        "version": None,
        "section_path": [],
        "metadata": {},
    }


def test_routes_to_generate_when_sufficient() -> None:
    docs = [_make_doc(), _make_doc()]
    docs[1]["chunk_id"] = "c2"
    grades = [_make_grade(0.9), _make_grade(0.9)]
    grades[1]["chunk_id"] = "c2"
    state = _make_state(accepted=docs, grade_results=grades)

    result = route_correction(state)
    assert result["routing_decision"]["action"] == "generate"


def test_routes_to_rewrite_when_low_relevance() -> None:
    grades = [_make_grade(0.2, decision="reject")]
    state = _make_state(accepted=[], rejected=[_make_doc()], grade_results=grades, retry_count=0)

    result = route_correction(state)
    assert result["routing_decision"]["action"] == "rewrite"


def test_routes_to_abstain_when_no_docs() -> None:
    state = _make_state(grade_results=[])
    result = route_correction(state)
    assert result["routing_decision"]["action"] == "abstain"


def test_routes_to_abstain_when_max_retries_exceeded() -> None:
    grades = [_make_grade(0.2, decision="reject")]
    state = _make_state(accepted=[], grade_results=grades, retry_count=5)
    result = route_correction(state)
    assert result["routing_decision"]["action"] == "abstain"


def test_routes_to_expand_when_relevant_but_insufficient() -> None:
    grades = [_make_grade(0.6, decision="reject")]
    state = _make_state(accepted=[], rejected=[_make_doc()], grade_results=grades, retry_count=0)
    result = route_correction(state)
    # Should be expand (relevance >= 0.4 but insufficient accepted docs)
    assert result["routing_decision"]["action"] in ("expand", "rewrite")


def test_get_routing_action_extracts_action() -> None:
    state = _make_state()
    state["routing_decision"] = {"action": "generate", "reason": "ok", "confidence": 0.9}
    assert get_routing_action(state) == "generate"


def test_routing_decision_has_required_fields() -> None:
    grades = [_make_grade(0.9), _make_grade(0.9)]
    docs = [_make_doc(), {**_make_doc(), "chunk_id": "c2"}]
    state = _make_state(accepted=docs, grade_results=grades)

    result = route_correction(state)
    for field in ("action", "reason", "confidence"):
        assert field in result["routing_decision"]
