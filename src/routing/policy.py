from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from src.graph.state import RAGState, RoutingDecision

logger = logging.getLogger(__name__)

_ROUTING_CONFIG_PATH = os.getenv("ROUTING_CONFIG", "configs/routing.yaml")

_DEFAULTS = {
    "generate_min_accepted_docs": 2,
    "generate_min_avg_relevance": 0.6,
    "rewrite_max_retries": 2,
    "expand_min_relevance": 0.4,
    "version_mismatch_threshold": 0.3,
    "abstain_when_no_accepted": True,
    "relevance_weight": 0.4,
    "sufficiency_weight": 0.3,
    "specificity_weight": 0.2,
    "version_match_weight": 0.1,
}


def _load_config() -> dict[str, Any]:
    try:
        with open(_ROUTING_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
            return data.get("routing", _DEFAULTS)
    except FileNotFoundError:
        return _DEFAULTS


def _composite_score(grade: dict[str, Any], cfg: dict[str, Any]) -> float:
    return (
        grade["relevance"] * cfg["relevance_weight"]
        + grade["sufficiency"] * cfg["sufficiency_weight"]
        + grade["specificity"] * cfg["specificity_weight"]
        + grade["version_match"] * cfg["version_match_weight"]
    )


def route_correction(state: RAGState) -> dict:
    accepted = state.get("accepted_docs", [])
    rejected = state.get("rejected_docs", [])
    grade_results = state.get("grade_results", [])
    metrics = dict(state.get("metrics", {}))
    trace = list(state.get("trace", []))

    cfg = _load_config()
    retry_count = metrics.get("retry_count", 0)
    max_retries = cfg["rewrite_max_retries"]

    if retry_count >= max_retries and not accepted:
        decision = RoutingDecision(
            action="abstain",
            reason=f"Exceeded max retries ({max_retries}) with no accepted docs",
            confidence=1.0,
        )
        trace.append("route_correction → abstain (max retries exceeded)")
        return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}

    if not grade_results and cfg["abstain_when_no_accepted"]:
        decision = RoutingDecision(
            action="abstain",
            reason="No documents retrieved or graded",
            confidence=1.0,
        )
        trace.append("route_correction → abstain (no docs)")
        return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}

    avg_relevance = sum(g["relevance"] for g in grade_results) / len(grade_results) if grade_results else 0.0
    avg_version_match = sum(g["version_match"] for g in grade_results) / len(grade_results) if grade_results else 1.0
    avg_composite = sum(_composite_score(g, cfg) for g in grade_results) / len(grade_results) if grade_results else 0.0

    n_accepted = len(accepted)
    min_accepted = cfg["generate_min_accepted_docs"]
    min_relevance = cfg["generate_min_avg_relevance"]

    if n_accepted >= min_accepted and avg_relevance >= min_relevance:
        decision = RoutingDecision(
            action="generate",
            reason=f"Got {n_accepted} good docs (avg relevance {avg_relevance:.2f})",
            confidence=min(avg_composite, 1.0),
        )
        trace.append(f"route_correction → generate ({n_accepted} docs, relevance={avg_relevance:.2f})")
        return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}

    if avg_version_match < cfg["version_mismatch_threshold"] and retry_count < max_retries:
        decision = RoutingDecision(
            action="alternate",
            reason=f"Chunks are wrong version (avg match={avg_version_match:.2f})",
            confidence=0.7,
        )
        trace.append(f"route_correction → alternate (version mismatch {avg_version_match:.2f})")
        return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}

    if avg_relevance >= cfg["expand_min_relevance"] and n_accepted < min_accepted and retry_count < 1:
        decision = RoutingDecision(
            action="expand",
            reason=f"Chunks are relevant ({avg_relevance:.2f}) but need more context — pulling parent sections",
            confidence=0.65,
        )
        trace.append(f"route_correction → expand (relevance={avg_relevance:.2f}, accepted={n_accepted})")
        return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}

    if retry_count < max_retries:
        decision = RoutingDecision(
            action="rewrite",
            reason=f"Low relevance ({avg_relevance:.2f}) — trying a different query",
            confidence=0.6,
        )
        trace.append(f"route_correction → rewrite (relevance={avg_relevance:.2f}, retry={retry_count})")
        return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}

    if accepted:
        decision = RoutingDecision(
            action="generate",
            reason=f"Generating with {n_accepted} doc(s) after corrections ran out",
            confidence=0.4,
        )
    else:
        decision = RoutingDecision(
            action="abstain",
            reason="Nothing usable found after all corrective attempts",
            confidence=0.9,
        )

    trace.append(f"route_correction → {decision.action} (fallback)")
    logger.info("Routing decision: %s — %s", decision.action, decision.reason)
    return {"routing_decision": decision.model_dump(), "trace": trace, "metrics": metrics}


def get_routing_action(state: RAGState) -> str:
    return state.get("routing_decision", {}).get("action", "abstain")
