"""Unit tests for document grader."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.grading.document_grader import grade_documents


def _make_state(retrieved: list[dict]) -> dict:
    return {
        "query": "How do I create a charge?",
        "rewritten_query": "",
        "query_type": "how_to",
        "constraints": {},
        "retrieved_docs": retrieved,
        "accepted_docs": [],
        "rejected_docs": [],
        "grade_results": [],
        "routing_decision": {},
        "answer": "",
        "citations": [],
        "verification_result": {},
        "trace": [],
        "metrics": {},
    }


def _make_doc(chunk_id: str, text: str = "Stripe charge docs") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc_{chunk_id}",
        "text": text,
        "source_url": "https://stripe.com/docs/api/charges",
        "score": 0.8,
        "version": None,
        "section_path": ["API Reference", "Charges"],
        "metadata": {},
    }


def _mock_grade_response(decision: str, relevance: float = 0.8) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = json.dumps({
        "relevance": relevance,
        "sufficiency": 0.7,
        "specificity": 0.8,
        "version_match": 1.0,
        "decision": decision,
        "rationale": "test rationale",
    })
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]
    mock_msg.usage.input_tokens = 100
    mock_msg.usage.output_tokens = 50
    return mock_msg


def test_grade_accepts_relevant_docs() -> None:
    docs = [_make_doc("c1"), _make_doc("c2")]
    state = _make_state(docs)

    with patch("src.grading.document_grader.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_grade_response("accept", relevance=0.9)

        result = grade_documents(state)

    assert len(result["accepted_docs"]) == 2
    assert len(result["rejected_docs"]) == 0
    assert len(result["grade_results"]) == 2


def test_grade_rejects_irrelevant_docs() -> None:
    docs = [_make_doc("c1"), _make_doc("c2")]
    state = _make_state(docs)

    with patch("src.grading.document_grader.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_grade_response("reject", relevance=0.2)

        result = grade_documents(state)

    assert len(result["rejected_docs"]) == 2
    assert len(result["accepted_docs"]) == 0


def test_grade_empty_docs() -> None:
    state = _make_state([])
    result = grade_documents(state)
    assert result["grade_results"] == []
    assert result["accepted_docs"] == []
    assert result["rejected_docs"] == []


def test_grade_results_have_required_fields() -> None:
    docs = [_make_doc("c1")]
    state = _make_state(docs)

    with patch("src.grading.document_grader.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_grade_response("accept")

        result = grade_documents(state)

    gr = result["grade_results"][0]
    for field in ["relevance", "sufficiency", "specificity", "version_match", "decision", "rationale"]:
        assert field in gr
