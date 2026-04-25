from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.graph.state import RAGState

logger = logging.getLogger(__name__)

_TRACES_PATH = os.getenv("TRACES_PATH", "data/processed/traces.jsonl")

_COST_PER_1K_INPUT = 0.003   # claude-sonnet-4-6 approximate
_COST_PER_1K_OUTPUT = 0.015


def _compute_token_cost(metrics: dict[str, Any]) -> float:
    input_tok = metrics.get("input_tokens", 0)
    output_tok = metrics.get("output_tokens", 0)
    return round(
        (input_tok / 1000) * _COST_PER_1K_INPUT + (output_tok / 1000) * _COST_PER_1K_OUTPUT,
        6,
    )


def log_metrics(state: RAGState) -> dict:
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    start_time = metrics.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 1)
    token_cost = _compute_token_cost(metrics)

    routing = state.get("routing_decision", {})
    verification = state.get("verification_result", {})

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": state.get("query", ""),
        "query_type": state.get("query_type", ""),
        "routing_action": routing.get("action", ""),
        "routing_reason": routing.get("reason", ""),
        "retry_count": metrics.get("retry_count", 0),
        "node_path": trace,
        "n_retrieved": len(state.get("retrieved_docs", [])),
        "n_accepted": len(state.get("accepted_docs", [])),
        "n_rejected": len(state.get("rejected_docs", [])),
        "answer_grounded": len(state.get("citations", [])) > 0,
        "abstained": "abstain" in routing.get("action", ""),
        "verification_verdict": verification.get("verdict", ""),
        "n_unsupported_claims": len(verification.get("unsupported_claims", [])),
        "latency_ms": latency_ms,
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "token_cost_usd": token_cost,
    }

    traces_path = Path(_TRACES_PATH)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    metrics["latency_ms"] = latency_ms
    metrics["token_cost_usd"] = token_cost
    trace.append(f"log_metrics → latency={latency_ms}ms, cost=${token_cost:.4f}")

    logger.info(
        "%s query took %dms, routed to %s, cost $%.4f",
        record["query_type"],
        latency_ms,
        record["routing_action"],
        token_cost,
    )

    return {"metrics": metrics, "trace": trace}
