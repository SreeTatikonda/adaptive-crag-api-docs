"""FastAPI service: POST /query, POST /feedback, GET /metrics, GET /health."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive CRAG — Stripe API Docs QA",
    description="Hierarchy-aware, corrective RAG for Stripe developer documentation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    version: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    grounded: bool
    abstained: bool
    query_type: str
    routing_action: str
    trace: list[str]
    latency_ms: float
    token_cost_usd: float


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int  # 1–5
    comment: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    from src.graph.workflow import run_query

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    start = time.time()
    try:
        state = run_query(req.query, version=req.version)
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail=str(exc))

    metrics = state.get("metrics", {})
    routing = state.get("routing_decision", {})
    latency_ms = metrics.get("latency_ms", round((time.time() - start) * 1000, 1))

    return QueryResponse(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        grounded=len(state.get("citations", [])) > 0,
        abstained=routing.get("action") == "abstain",
        query_type=state.get("query_type", ""),
        routing_action=routing.get("action", ""),
        trace=state.get("trace", []),
        latency_ms=latency_ms,
        token_cost_usd=metrics.get("token_cost_usd", 0.0),
    )


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest) -> dict[str, str]:
    feedback_path = Path(os.getenv("FEEDBACK_PATH", "data/processed/feedback.jsonl"))
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": req.query,
        "answer": req.answer,
        "rating": req.rating,
        "comment": req.comment,
    }
    with open(feedback_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "recorded"}


@app.get("/metrics")
def metrics_endpoint() -> dict[str, Any]:
    traces_path = Path(os.getenv("TRACES_PATH", "data/processed/traces.jsonl"))
    if not traces_path.exists():
        return {"total_queries": 0}

    records = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
    if not records:
        return {"total_queries": 0}

    latencies = [r.get("latency_ms", 0) for r in records]
    costs = [r.get("token_cost_usd", 0) for r in records]
    routing_actions = [r.get("routing_action", "") for r in records]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "total_queries": n,
        "grounded_rate": round(sum(r.get("answer_grounded", False) for r in records) / n, 3),
        "abstention_rate": round(sum("abstain" in r.get("routing_action", "") for r in records) / n, 3),
        "correction_rate": round(sum(r.get("routing_action", "") != "generate" for r in records) / n, 3),
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "p50_latency_ms": sorted_latencies[n // 2],
        "p95_latency_ms": sorted_latencies[int(n * 0.95)],
        "avg_token_cost_usd": round(sum(costs) / n, 5),
        "routing_breakdown": {
            action: routing_actions.count(action)
            for action in set(routing_actions)
        },
    }
