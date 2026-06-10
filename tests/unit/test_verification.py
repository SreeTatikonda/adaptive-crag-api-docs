"""Unit tests for verification safety behavior."""

from unittest.mock import MagicMock, patch

from src.verification.claim_verifier import verify_answer


def make_state() -> dict:
    return {
        "answer": "Use the foo parameter to create a PaymentIntent.",
        "accepted_docs": [
            {
                "chunk_id": "chunk_1",
                "text": "PaymentIntent creation requires amount and currency.",
            }
        ],
        "trace": [],
        "metrics": {},
    }


@patch("src.verification.claim_verifier.anthropic.Anthropic")
def test_verifier_parse_failure_abstains(mock_anthropic: MagicMock) -> None:
    client = MagicMock()
    mock_anthropic.return_value = client

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="not valid json")]
    fake_response.usage.input_tokens = 10
    fake_response.usage.output_tokens = 5

    client.messages.create.return_value = fake_response

    result = verify_answer(make_state())

    assert result["verification_result"]["verdict"] == "abstain"
    assert result["verification_result"]["supported"] is False
    assert result["citations"] == []
    assert result["answer"].startswith("I could not verify")