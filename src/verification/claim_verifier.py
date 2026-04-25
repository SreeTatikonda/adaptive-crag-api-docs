from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from src.graph.state import RAGState

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
Read this answer and the documentation chunks it was built from.
Find any claims in the answer that aren't actually backed by the chunks — specific API behavior, parameter names, version details, or anything that contradicts what the chunks say.

Return JSON:
{
  "supported": true or false,
  "unsupported_claims": ["<claim>", ...],
  "verdict": "pass" or "revise" or "abstain",
  "explanation": "<what you found>"
}

"pass" means everything checks out.
"revise" means there are small issues but the answer is mostly fine.
"abstain" means there are significant claims that aren't in the docs — don't return this answer.
"""


def _build_verification_prompt(answer: str, accepted_docs: list[dict[str, Any]]) -> str:
    parts = []
    for doc in accepted_docs[:5]:
        parts.append(f"Chunk [{doc['chunk_id']}]:\n{doc['text'][:1500]}")
    context = "\n\n---\n\n".join(parts)
    return f"Answer:\n{answer}\n\nSource chunks:\n\n{context}"


def verify_answer(state: RAGState) -> dict:
    answer = state.get("answer", "")
    accepted = state.get("accepted_docs", [])
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    if not answer or not accepted:
        result = {
            "supported": True,
            "unsupported_claims": [],
            "verdict": "pass",
            "explanation": "Nothing to verify",
        }
        trace.append("verify_answer → skipped")
        return {"verification_result": result, "trace": trace, "metrics": metrics}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = _build_verification_prompt(answer, accepted)

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0,
        system=_VERIFY_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    metrics["input_tokens"] = metrics.get("input_tokens", 0) + msg.usage.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + msg.usage.output_tokens

    try:
        result: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "supported": True,
            "unsupported_claims": [],
            "verdict": "pass",
            "explanation": "Could not parse verifier response",
        }

    verdict = result.get("verdict", "pass")
    n_unsupported = len(result.get("unsupported_claims", []))

    trace.append(f"verify_answer → {verdict} ({n_unsupported} unsupported claims)")
    logger.info("Verification verdict: %s, unsupported claims: %d", verdict, n_unsupported)

    if verdict == "abstain":
        return {
            "answer": "I couldn't verify this answer against the available docs. Check stripe.com/docs for accurate information.",
            "verification_result": result,
            "trace": trace,
            "metrics": metrics,
        }

    return {"verification_result": result, "trace": trace, "metrics": metrics}
