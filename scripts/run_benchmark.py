#!/usr/bin/env python
"""Run evaluation harness comparing NaiveRAG, HybridRAG, StaticCRAG, AdaptiveCRAG."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer()


@app.command()
def main(
    benchmark_path: Path = typer.Option(
        Path("data/benchmarks/stripe_qa.json"), help="Benchmark QA dataset"
    ),
    results_dir: Path = typer.Option(
        Path("data/processed/eval_results"), help="Where to save evaluation results"
    ),
    baselines: str = typer.Option(
        "naive_rag,hybrid_rag,static_crag,adaptive_crag",
        help="Comma-separated list of baselines to run",
    ),
) -> None:
    from src.evaluation.harness import EvaluationHarness

    if not benchmark_path.exists():
        typer.echo(f"Benchmark file not found: {benchmark_path}", err=True)
        typer.echo("Creating sample benchmark…")
        _create_sample_benchmark(benchmark_path)

    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_list = [b.strip() for b in baselines.split(",")]

    typer.echo(f"Running benchmark: {benchmark_path}")
    typer.echo(f"Baselines: {baseline_list}")

    harness = EvaluationHarness(benchmark_path=str(benchmark_path), results_dir=str(results_dir))
    report = harness.run(baselines=baseline_list)

    report_path = results_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    typer.echo(f"\nReport saved to {report_path}")

    typer.echo("\n=== Summary ===")
    for baseline, metrics in report.items():
        typer.echo(f"\n{baseline}:")
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")


def _create_sample_benchmark(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = [
        {
            "id": "q001",
            "query": "How do I create a PaymentIntent in Python?",
            "query_type": "how_to",
            "reference_answer": "Use stripe.PaymentIntent.create() with amount and currency parameters.",
            "relevant_doc_ids": [],
            "api_version": None,
        },
        {
            "id": "q002",
            "query": "What HTTP method does the Charges API use to create a charge?",
            "query_type": "fact_lookup",
            "reference_answer": "POST /v1/charges",
            "relevant_doc_ids": [],
            "api_version": None,
        },
        {
            "id": "q003",
            "query": "What does error code card_declined mean?",
            "query_type": "error_debugging",
            "reference_answer": "The card was declined by the issuer.",
            "relevant_doc_ids": [],
            "api_version": None,
        },
        {
            "id": "q004",
            "query": "How do I set up webhooks to listen for payment_intent.succeeded?",
            "query_type": "how_to",
            "reference_answer": "Create a webhook endpoint and register it in the Stripe dashboard.",
            "relevant_doc_ids": [],
            "api_version": None,
        },
        {
            "id": "q005",
            "query": "What is the difference between a Customer and a PaymentMethod?",
            "query_type": "fact_lookup",
            "reference_answer": "A Customer is a persistent record; a PaymentMethod stores payment details.",
            "relevant_doc_ids": [],
            "api_version": None,
        },
    ]
    path.write_text(json.dumps(sample, indent=2))
    typer.echo(f"Sample benchmark created at {path}")


if __name__ == "__main__":
    app()
