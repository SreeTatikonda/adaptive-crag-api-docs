"""Evaluation harness: run baseline comparisons and compute all metrics."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    compute_retrieval_metrics,
    llm_judge_answer,
    citation_correctness,
    abstention_accuracy,
)

logger = logging.getLogger(__name__)


class EvaluationHarness:
    def __init__(self, benchmark_path: str, results_dir: str) -> None:
        self.benchmark_path = benchmark_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(benchmark_path) as f:
            self.questions: list[dict] = json.load(f)

    def _run_adaptive_crag(self, query: str, version: str | None) -> dict[str, Any]:
        from src.graph.workflow import run_query
        return run_query(query, version=version)

    def _run_naive_rag(self, query: str) -> dict[str, Any]:
        """Simple retrieve-then-generate without grading or correction."""
        from src.indexing.chroma_index import load_chroma_collection, query_chroma
        from src.generation.answer_generator import generate_answer as _gen

        try:
            collection = load_chroma_collection()
            docs = query_chroma(collection, query, top_k=5)
        except Exception:
            docs = []

        fake_state: Any = {
            "query": query,
            "rewritten_query": "",
            "accepted_docs": docs,
            "routing_decision": {"action": "generate"},
            "trace": ["naive_rag → retrieve → generate"],
            "metrics": {"start_time": time.time()},
            "retrieved_docs": docs,
            "grade_results": [],
            "rejected_docs": [],
            "citations": [],
            "verification_result": {},
            "constraints": {},
            "query_type": "fact_lookup",
        }
        result = _gen(fake_state)
        return {**fake_state, **result}

    def _run_hybrid_rag(self, query: str) -> dict[str, Any]:
        """Dense + BM25 retrieval without grading."""
        from src.retrieval.hybrid import retrieve_hybrid
        from src.generation.answer_generator import generate_answer as _gen

        base_state: Any = {
            "query": query,
            "rewritten_query": "",
            "constraints": {},
            "retrieved_docs": [],
            "accepted_docs": [],
            "rejected_docs": [],
            "grade_results": [],
            "routing_decision": {"action": "generate"},
            "trace": ["hybrid_rag"],
            "metrics": {"start_time": time.time()},
            "citations": [],
            "verification_result": {},
            "query_type": "fact_lookup",
        }
        ret = retrieve_hybrid(base_state)
        state = {**base_state, **ret, "accepted_docs": ret.get("retrieved_docs", [])[:5]}
        result = _gen(state)
        return {**state, **result}

    def _run_static_crag(self, query: str) -> dict[str, Any]:
        """Retrieve + grade with fixed threshold, no adaptive routing."""
        from src.retrieval.hybrid import retrieve_hybrid
        from src.grading.document_grader import grade_documents
        from src.generation.answer_generator import generate_answer as _gen

        base_state: Any = {
            "query": query,
            "rewritten_query": "",
            "constraints": {},
            "retrieved_docs": [],
            "accepted_docs": [],
            "rejected_docs": [],
            "grade_results": [],
            "routing_decision": {"action": "generate"},
            "trace": ["static_crag"],
            "metrics": {"start_time": time.time()},
            "citations": [],
            "verification_result": {},
            "query_type": "fact_lookup",
        }
        ret = retrieve_hybrid(base_state)
        state = {**base_state, **ret}
        graded = grade_documents(state)
        state = {**state, **graded}
        state["routing_decision"] = {"action": "generate" if state["accepted_docs"] else "abstain"}
        result = _gen(state)
        return {**state, **result}

    def _evaluate_result(
        self, question: dict, state: dict[str, Any], latency_ms: float
    ) -> dict[str, Any]:
        query = question["query"]
        reference = question.get("reference_answer", "")
        relevant_ids = set(question.get("relevant_doc_ids", []))

        retrieved_ids = [d["chunk_id"] for d in state.get("retrieved_docs", [])]
        answer = state.get("answer", "")
        citations = state.get("citations", [])
        abstained = state.get("routing_decision", {}).get("action") == "abstain"
        accepted_ids = {d["chunk_id"] for d in state.get("accepted_docs", [])}

        retrieval_metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids) if relevant_ids else {}

        judge_scores: dict[str, float] = {}
        if reference and answer and not abstained:
            judge_scores = llm_judge_answer(query, reference, answer)

        return {
            "question_id": question.get("id", ""),
            "query_type": question.get("query_type", ""),
            "abstained": abstained,
            "latency_ms": latency_ms,
            "n_retrieved": len(retrieved_ids),
            "n_accepted": len(accepted_ids),
            "citation_correctness": citation_correctness(citations, accepted_ids),
            "abstention_accuracy": abstention_accuracy(abstained, bool(accepted_ids)),
            **retrieval_metrics,
            **judge_scores,
        }

    def _aggregate(self, results: list[dict]) -> dict[str, float]:
        if not results:
            return {}
        numeric_keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
        return {k: round(sum(r.get(k, 0) for r in results) / len(results), 4) for k in numeric_keys}

    def run(self, baselines: list[str] | None = None) -> dict[str, Any]:
        baselines = baselines or ["naive_rag", "hybrid_rag", "static_crag", "adaptive_crag"]
        report: dict[str, Any] = {}

        runners = {
            "naive_rag": lambda q: self._run_naive_rag(q["query"]),
            "hybrid_rag": lambda q: self._run_hybrid_rag(q["query"]),
            "static_crag": lambda q: self._run_static_crag(q["query"]),
            "adaptive_crag": lambda q: self._run_adaptive_crag(q["query"], q.get("api_version")),
        }

        for baseline in baselines:
            if baseline not in runners:
                logger.warning("Unknown baseline: %s", baseline)
                continue

            logger.info("Running baseline: %s", baseline)
            per_q: list[dict] = []

            for question in self.questions:
                start = time.time()
                try:
                    state = runners[baseline](question)
                    latency_ms = round((time.time() - start) * 1000, 1)
                    result = self._evaluate_result(question, state, latency_ms)
                except Exception as exc:
                    logger.error("Error on %s/%s: %s", baseline, question.get("id"), exc)
                    result = {"question_id": question.get("id", ""), "error": str(exc)}
                per_q.append(result)

            per_q_path = self.results_dir / f"{baseline}_per_question.json"
            per_q_path.write_text(json.dumps(per_q, indent=2))

            report[baseline] = self._aggregate([r for r in per_q if "error" not in r])

        return report
