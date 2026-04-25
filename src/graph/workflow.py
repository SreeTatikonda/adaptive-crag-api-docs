from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.state import RAGState
from src.query.classifier import classify_query
from src.query.constraints import extract_constraints
from src.retrieval.hybrid import (
    expand_context,
    rerank_documents,
    retrieve_hybrid,
    rewrite_query,
)
from src.grading.document_grader import grade_documents
from src.routing.policy import get_routing_action, route_correction
from src.generation.answer_generator import generate_answer
from src.verification.claim_verifier import verify_answer
from src.observability.logger import log_metrics


def _init_state(state: RAGState) -> dict[str, Any]:
    metrics = dict(state.get("metrics", {}))
    if "start_time" not in metrics:
        metrics["start_time"] = time.time()
    return {"metrics": metrics}


def build_graph() -> Any:
    builder: StateGraph = StateGraph(RAGState)

    builder.add_node("init", _init_state)
    builder.add_node("classify_query", classify_query)
    builder.add_node("extract_constraints", extract_constraints)
    builder.add_node("retrieve_hybrid", retrieve_hybrid)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("route_correction", route_correction)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("expand_context", expand_context)
    builder.add_node("rerank_documents", rerank_documents)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("verify_answer", verify_answer)
    builder.add_node("log_metrics", log_metrics)

    builder.add_edge(START, "init")
    builder.add_edge("init", "classify_query")
    builder.add_edge("classify_query", "extract_constraints")
    builder.add_edge("extract_constraints", "retrieve_hybrid")
    builder.add_edge("retrieve_hybrid", "grade_documents")
    builder.add_edge("grade_documents", "route_correction")

    builder.add_conditional_edges(
        "route_correction",
        get_routing_action,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "expand": "expand_context",
            "alternate": "retrieve_hybrid",
            "abstain": "generate_answer",
        },
    )

    builder.add_edge("rewrite_query", "retrieve_hybrid")
    builder.add_edge("expand_context", "rerank_documents")
    builder.add_edge("rerank_documents", "grade_documents")
    builder.add_edge("generate_answer", "verify_answer")
    builder.add_edge("verify_answer", "log_metrics")
    builder.add_edge("log_metrics", END)

    return builder.compile()


_GRAPH = None


def get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_query(query: str, version: str | None = None) -> dict[str, Any]:
    graph = get_graph()

    initial_state: RAGState = {
        "query": query,
        "query_type": "",
        "constraints": {"api_version": version} if version else {},
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

    return graph.invoke(initial_state)
