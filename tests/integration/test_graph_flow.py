"""Integration tests for the LangGraph workflow using mocked LLM and retrieval."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


def _mock_classify(query_type: str = "how_to"):
    mock_content = MagicMock()
    mock_content.text = json.dumps({"query_type": query_type, "confidence": 0.9, "reasoning": "test"})
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]
    mock_msg.usage.input_tokens = 50
    mock_msg.usage.output_tokens = 20
    return mock_msg


def _mock_constraints():
    mock_content = MagicMock()
    mock_content.text = json.dumps({"product": "PaymentIntents", "api_version": None, "language": "python", "endpoint": None, "auth_context": None, "sdk": None})
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]
    mock_msg.usage.input_tokens = 60
    mock_msg.usage.output_tokens = 30
    return mock_msg


def _mock_grade(decision: str = "accept", relevance: float = 0.85):
    mock_content = MagicMock()
    mock_content.text = json.dumps({
        "relevance": relevance,
        "sufficiency": 0.8,
        "specificity": 0.8,
        "version_match": 1.0,
        "decision": decision,
        "rationale": "test",
    })
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]
    mock_msg.usage.input_tokens = 100
    mock_msg.usage.output_tokens = 50
    return mock_msg


def _mock_generate():
    mock_content = MagicMock()
    mock_content.text = "Use stripe.PaymentIntent.create() [abc12345678901234] to create a payment intent."
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]
    mock_msg.usage.input_tokens = 300
    mock_msg.usage.output_tokens = 100
    return mock_msg


def _mock_verify():
    mock_content = MagicMock()
    mock_content.text = json.dumps({
        "supported": True,
        "unsupported_claims": [],
        "verdict": "pass",
        "explanation": "All claims are supported.",
    })
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]
    mock_msg.usage.input_tokens = 200
    mock_msg.usage.output_tokens = 60
    return mock_msg


def _make_mock_docs():
    return [
        {
            "chunk_id": "abc12345678901234",
            "doc_id": "doc001",
            "text": "stripe.PaymentIntent.create() creates a PaymentIntent in Stripe.",
            "source_url": "https://stripe.com/docs/api/payment_intents",
            "score": 0.9,
            "version": None,
            "section_path": ["API Reference", "PaymentIntents"],
            "metadata": {"parent_chunk_id": ""},
        },
        {
            "chunk_id": "def98765432109876",
            "doc_id": "doc001",
            "text": "PaymentIntent parameters include amount, currency, and payment_method.",
            "source_url": "https://stripe.com/docs/api/payment_intents",
            "score": 0.8,
            "version": None,
            "section_path": ["API Reference", "PaymentIntents"],
            "metadata": {"parent_chunk_id": "abc12345678901234"},
        },
    ]


@pytest.fixture
def mock_retrieval(monkeypatch):
    docs = _make_mock_docs()

    def fake_retrieve_hybrid(state):
        return {
            "retrieved_docs": docs,
            "trace": list(state.get("trace", [])) + ["retrieve_hybrid → 2 docs"],
            "metrics": state.get("metrics", {}),
        }

    monkeypatch.setattr("src.retrieval.hybrid.retrieve_hybrid", fake_retrieve_hybrid)
    return docs


@patch("src.query.classifier.anthropic.Anthropic")
@patch("src.query.constraints.anthropic.Anthropic")
@patch("src.grading.document_grader.anthropic.Anthropic")
@patch("src.generation.answer_generator.anthropic.Anthropic")
@patch("src.verification.claim_verifier.anthropic.Anthropic")
def test_full_graph_generate_path(
    mock_verifier_cls,
    mock_gen_cls,
    mock_grader_cls,
    mock_constraints_cls,
    mock_classifier_cls,
    mock_retrieval,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TRACES_PATH", str(tmp_path / "traces.jsonl"))

    for cls, resp in [
        (mock_classifier_cls, _mock_classify()),
        (mock_constraints_cls, _mock_constraints()),
        (mock_verifier_cls, _mock_verify()),
        (mock_gen_cls, _mock_generate()),
    ]:
        mock_client = MagicMock()
        cls.return_value = mock_client
        mock_client.messages.create.return_value = resp

    grader_client = MagicMock()
    mock_grader_cls.return_value = grader_client
    grader_client.messages.create.return_value = _mock_grade("accept", 0.85)

    from src.graph.workflow import run_query

    result = run_query("How do I create a PaymentIntent in Python?")

    assert result["query_type"] == "how_to"
    assert result["answer"] != ""
    assert result["routing_decision"]["action"] == "generate"
    assert len(result["trace"]) > 0


@patch("src.query.classifier.anthropic.Anthropic")
@patch("src.query.constraints.anthropic.Anthropic")
@patch("src.grading.document_grader.anthropic.Anthropic")
@patch("src.generation.answer_generator.anthropic.Anthropic")
@patch("src.verification.claim_verifier.anthropic.Anthropic")
def test_graph_abstain_path(
    mock_verifier_cls,
    mock_gen_cls,
    mock_grader_cls,
    mock_constraints_cls,
    mock_classifier_cls,
    mock_retrieval,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TRACES_PATH", str(tmp_path / "traces.jsonl"))

    for cls, resp in [
        (mock_classifier_cls, _mock_classify("out_of_scope")),
        (mock_constraints_cls, _mock_constraints()),
    ]:
        mock_client = MagicMock()
        cls.return_value = mock_client
        mock_client.messages.create.return_value = resp

    grader_client = MagicMock()
    mock_grader_cls.return_value = grader_client
    grader_client.messages.create.return_value = _mock_grade("reject", 0.1)

    abstain_content = MagicMock()
    abstain_content.text = "I cannot find sufficient documentation to answer this question."
    abstain_msg = MagicMock()
    abstain_msg.content = [abstain_content]
    abstain_msg.usage.input_tokens = 50
    abstain_msg.usage.output_tokens = 30
    gen_client = MagicMock()
    mock_gen_cls.return_value = gen_client
    gen_client.messages.create.return_value = abstain_msg

    verify_client = MagicMock()
    mock_verifier_cls.return_value = verify_client
    verify_client.messages.create.return_value = _mock_verify()

    from src.graph.workflow import run_query

    result = run_query("What is the meaning of life?")
    assert result["answer"] != ""
