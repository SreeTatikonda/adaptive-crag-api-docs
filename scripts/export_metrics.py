#!/usr/bin/env python
"""Export traces.jsonl to CSV and summary JSON for analysis."""

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
    traces_path: Path = typer.Option(
        Path("data/processed/traces.jsonl"), help="Path to traces JSONL file"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed/exports"), help="Export output directory"
    ),
) -> None:
    if not traces_path.exists():
        typer.echo(f"No traces file found at {traces_path}", err=True)
        raise typer.Exit(1)

    records = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
    if not records:
        typer.echo("No records found in traces file.")
        raise typer.Exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export CSV
    import csv
    csv_path = output_dir / "traces.csv"
    keys = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            flat = {k: ("|".join(v) if isinstance(v, list) else v) for k, v in r.items()}
            writer.writerow(flat)
    typer.echo(f"CSV exported → {csv_path}")

    # Summary stats
    n = len(records)
    latencies = sorted(r.get("latency_ms", 0) for r in records)
    costs = [r.get("token_cost_usd", 0) for r in records]
    routing_actions = [r.get("routing_action", "") for r in records]

    summary = {
        "total_queries": n,
        "grounded_rate": round(sum(r.get("answer_grounded", False) for r in records) / n, 3),
        "abstention_rate": round(sum("abstain" in r.get("routing_action", "") for r in records) / n, 3),
        "correction_rate": round(sum(r.get("routing_action", "") != "generate" for r in records) / n, 3),
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "p50_latency_ms": latencies[n // 2],
        "p95_latency_ms": latencies[int(n * 0.95)],
        "total_cost_usd": round(sum(costs), 4),
        "avg_cost_usd": round(sum(costs) / n, 5),
        "routing_breakdown": {a: routing_actions.count(a) for a in set(routing_actions)},
        "query_type_breakdown": {
            qt: sum(r.get("query_type", "") == qt for r in records)
            for qt in set(r.get("query_type", "") for r in records)
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"Summary exported → {summary_path}")

    typer.echo("\n=== Metrics Summary ===")
    for k, v in summary.items():
        if not isinstance(v, dict):
            typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
