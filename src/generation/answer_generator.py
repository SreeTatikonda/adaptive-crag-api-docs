from __future__ import annotations

import logging
import os
import re
from typing import Any

import anthropic

from src.graph.state import AnswerResult, RAGState

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """\
Answer the developer's question using only the documentation chunks provided below.
Cite every chunk you use by putting its ID in brackets inline, like: "use the `amount` parameter [abc123]".
If the chunks don't fully cover the question, say so — don't fill in gaps from memory.
Use markdown for code, endpoints, and parameters. Keep it concise and accurate.
"""

_NO_EVIDENCE_PROMPT = """\
The documentation search didn't turn up enough to answer this question confidently.
Write a short honest response explaining what you looked for and where the developer might find a better answer (stripe.com/docs or Stripe support).
"""


def _build_context(accepted_docs: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for doc in accepted_docs:
        chunk_id = doc["chunk_id"]
        section = " > ".join(doc.get("section_path", []))
        url = doc.get("source_url", "")
        text = doc["text"]
        parts.append(f"[{chunk_id}] Section: {section}\nURL: {url}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(answer_text: str, accepted_docs: list[dict[str, Any]]) -> list[str]:
    referenced_ids = set(re.findall(r"\[([a-f0-9]{16,20})\]", answer_text))
    valid_ids = {d["chunk_id"] for d in accepted_docs}
    cited = referenced_ids & valid_ids
    citations = []
    for doc in accepted_docs:
        if doc["chunk_id"] in cited:
            citations.append(f"[{doc['chunk_id']}] {doc.get('source_url', '')}")
    return citations


def generate_answer(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    accepted = state.get("accepted_docs", [])
    routing = state.get("routing_decision", {})
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if routing.get("action") == "abstain" or not accepted:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            temperature=0.3,
            system=_NO_EVIDENCE_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        answer_text = msg.content[0].text.strip()
        result = AnswerResult(answer=answer_text, citations=[], grounded=False, abstained=True)
        metrics["input_tokens"] = metrics.get("input_tokens", 0) + msg.usage.input_tokens
        metrics["output_tokens"] = metrics.get("output_tokens", 0) + msg.usage.output_tokens
        trace.append("generate_answer → abstained")
        return {
            "answer": result.answer,
            "citations": result.citations,
            "trace": trace,
            "metrics": metrics,
        }

    context = _build_context(accepted)
    user_message = f"Question: {query}\n\nDocumentation chunks:\n\n{context}"

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        temperature=0.3,
        system=_ANSWER_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = msg.content[0].text.strip()
    citations = _extract_citations(answer_text, accepted)
    grounded = len(citations) > 0

    metrics["input_tokens"] = metrics.get("input_tokens", 0) + msg.usage.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + msg.usage.output_tokens

    trace.append(f"generate_answer → {len(answer_text)} chars, {len(citations)} citations, grounded={grounded}")
    logger.info("Generated answer: %d chars, %d citations", len(answer_text), len(citations))

    return {
        "answer": answer_text,
        "citations": citations,
        "trace": trace,
        "metrics": metrics,
    }
