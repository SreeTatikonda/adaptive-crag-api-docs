from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    source_url: str
    score: float
    version: str | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    doc_id: str
    chunk_id: str
    relevance: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    version_match: float = Field(ge=0.0, le=1.0)
    decision: str  # "accept" | "reject"
    rationale: str


class RoutingDecision(BaseModel):
    action: str  # "generate" | "rewrite" | "expand" | "alternate" | "abstain"
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnswerResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    grounded: bool = False
    abstained: bool = False


class RAGState(TypedDict):
    query: str
    query_type: str
    constraints: dict[str, Any]
    retrieved_docs: list[dict]
    accepted_docs: list[dict]
    rejected_docs: list[dict]
    grade_results: list[dict]
    routing_decision: dict[str, Any]
    rewritten_query: str
    answer: str
    citations: list[str]
    verification_result: dict[str, Any]
    trace: list[str]
    metrics: dict[str, Any]
