#!/usr/bin/env python
"""Scrape Stripe docs and normalize to structured JSON."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.stripe_scraper import scrape_all, save_raw
from src.ingestion.normalizer import normalize_all, save_processed

app = typer.Typer()


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Directory for raw scraped JSON"),
    processed_dir: Path = typer.Option(Path("data/processed/docs"), help="Directory for normalized docs"),
    delay: float = typer.Option(0.5, help="Delay between HTTP requests in seconds"),
) -> None:
    typer.echo("Step 1/2: Scraping Stripe docs…")
    docs = scrape_all(delay=delay)
    save_raw(docs, raw_dir)
    typer.echo(f"Scraped {len(docs)} pages → {raw_dir}")

    typer.echo("Step 2/2: Normalizing docs…")
    normalized = normalize_all(raw_dir)
    save_processed(normalized, processed_dir)
    typer.echo(f"Normalized {len(normalized)} docs → {processed_dir}")


if __name__ == "__main__":
    app()
