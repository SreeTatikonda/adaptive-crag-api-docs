#!/usr/bin/env python
"""Build ChromaDB vector index and BM25 lexical index from processed docs."""

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

from src.chunking.hierarchy_chunker import chunk_corpus
from src.indexing.chroma_index import build_chroma_index
from src.indexing.bm25_index import build_bm25_index

app = typer.Typer()


@app.command()
def main(
    docs_dir: Path = typer.Option(Path("data/processed/docs"), help="Directory of normalized docs"),
    chroma_dir: str = typer.Option("data/processed/chroma", help="ChromaDB persist directory"),
    bm25_path: str = typer.Option("data/processed/bm25_index.pkl", help="BM25 index output path"),
    chunk_size: int = typer.Option(512, help="Chunk size in tokens"),
    chunk_overlap: int = typer.Option(64, help="Overlap tokens between chunks"),
) -> None:
    typer.echo(f"Loading docs from {docs_dir}…")
    docs = []
    for path in sorted(docs_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        docs.append(json.loads(path.read_text()))
    typer.echo(f"Loaded {len(docs)} docs")

    typer.echo("Chunking docs (hierarchy-aware)…")
    chunks = chunk_corpus(docs, chunk_size=chunk_size, overlap=chunk_overlap)
    typer.echo(f"Created {len(chunks)} chunks")

    typer.echo("Building ChromaDB index…")
    collection = build_chroma_index(chunks, persist_dir=chroma_dir)
    typer.echo(f"ChromaDB: {collection.count()} chunks indexed")

    typer.echo("Building BM25 index…")
    build_bm25_index(chunks, save_path=bm25_path)
    typer.echo(f"BM25 index saved to {bm25_path}")

    typer.echo("All indices built successfully.")


if __name__ == "__main__":
    app()
