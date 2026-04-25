from __future__ import annotations

import json
import logging
import os

import anthropic

from src.graph.state import RAGState

logger = logging.getLogger(__name__)

QUERY_TYPES = ["fact_lookup", "how_to", "error_debugging", "migration", "sdk_usage", "out_of_scope"]

_CLASSIFY_PROMPT = """\
Look at this developer question and pick the single best category for it:

- fact_lookup: wants a specific fact, parameter, field, or return value
- how_to: wants to know how to accomplish something
- error_debugging: asking about an error code, exception, or broken behavior
- migration: moving between API versions or switching to a newer approach
- sdk_usage: question is specifically about a client library or SDK
- out_of_scope: nothing to do with API docs

Return JSON: {"query_type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<why>"}
"""


def classify_query(state: RAGState) -> dict:
    query = state["query"]
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0,
        system=_CLASSIFY_PROMPT,
        messages=[{"role": "user", "content": query}],
    )

    raw = message.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        query_type = parsed.get("query_type", "fact_lookup")
        if query_type not in QUERY_TYPES:
            query_type = "fact_lookup"
    except (json.JSONDecodeError, KeyError):
        query_type = "fact_lookup"

    metrics["input_tokens"] = metrics.get("input_tokens", 0) + message.usage.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + message.usage.output_tokens
    trace.append(f"classify_query → {query_type}")
    logger.info("Query classified as: %s", query_type)

    return {"query_type": query_type, "trace": trace, "metrics": metrics}
