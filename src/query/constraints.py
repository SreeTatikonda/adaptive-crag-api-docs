from __future__ import annotations

import json
import logging
import os

import anthropic

from src.graph.state import RAGState

logger = logging.getLogger(__name__)

_CONSTRAINTS_PROMPT = """\
Pull out any Stripe-specific constraints mentioned in this developer question.
If something isn't mentioned, use null — don't guess.

Fields to extract:
- product: which Stripe product (e.g. "Charges", "PaymentIntents", "Subscriptions", "Customers")
- api_version: a date-formatted version string like "2024-06-20"
- language: programming language or runtime (e.g. "python", "node", "ruby", "curl")
- endpoint: a literal API path (e.g. "/v1/charges")
- auth_context: anything about auth (e.g. "secret key", "publishable key", "webhook secret")
- sdk: a specific SDK name (e.g. "stripe-python", "stripe-node")

Return only JSON matching those field names.
"""


def extract_constraints(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0,
        system=_CONSTRAINTS_PROMPT,
        messages=[{"role": "user", "content": query}],
    )

    raw = message.content[0].text.strip()
    try:
        constraints = json.loads(raw)
    except json.JSONDecodeError:
        constraints = {}

    metrics["input_tokens"] = metrics.get("input_tokens", 0) + message.usage.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + message.usage.output_tokens
    trace.append(f"extract_constraints → {constraints}")
    logger.info("Extracted constraints: %s", constraints)

    return {"constraints": constraints, "trace": trace, "metrics": metrics}
