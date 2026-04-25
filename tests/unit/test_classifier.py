"""Unit tests for query classifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.query.classifier import classify_query, QUERY_TYPES


def _make_state(query: str) -> dict:
    return {
        "query": query,
        "query_type": "",
        "constraints": {},
        "retrieved_docs": [],
        "accepted_docs": [],
        "rejected_docs": [],
        "grade_results": [],
        "routing_decision": {},
        "rewritten_query": "",
        "answer": "",
        "citations": [],
        "verification_result": {},
        "trace": [],
        "metrics": {},
    }


def _mock_claude_response(query_type: str) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = json.dumps({"query_type": query_type, "confidence": 0.9, "reasoning": "test"})
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_message.usage.input_tokens = 50
    mock_message.usage.output_tokens = 20
    return mock_message


@pytest.mark.parametrize("query,expected_type", [
    ("How do I create a PaymentIntent?", "how_to"),
    ("What does card_declined mean?", "error_debugging"),
    ("What is the endpoint for listing charges?", "fact_lookup"),
    ("How do I migrate from Charges to PaymentIntents?", "migration"),
    ("Show me Python SDK example for subscriptions", "sdk_usage"),
])
def test_classify_query_types(query: str, expected_type: str) -> None:
    state = _make_state(query)
    mock_msg = _mock_claude_response(expected_type)

    with patch("src.query.classifier.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_msg

        result = classify_query(state)

    assert result["query_type"] == expected_type
    assert result["query_type"] in QUERY_TYPES
    assert len(result["trace"]) == 1
    assert "classify_query" in result["trace"][0]


def test_classify_query_invalid_json_fallback() -> None:
    state = _make_state("some query")
    mock_content = MagicMock()
    mock_content.text = "not json"
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 5

    with patch("src.query.classifier.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_message

        result = classify_query(state)

    assert result["query_type"] == "fact_lookup"


def test_classify_updates_metrics() -> None:
    state = _make_state("test query")
    mock_msg = _mock_claude_response("how_to")

    with patch("src.query.classifier.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_msg

        result = classify_query(state)

    assert result["metrics"]["input_tokens"] == 50
    assert result["metrics"]["output_tokens"] == 20
