"""ChromaDB dense vector index with metadata filtering support."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

COLLECTION_NAME = "stripe_api_docs"


def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _get_embedding_fn(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def build_chroma_index(
    chunks: list[dict[str, Any]],
    persist_dir: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 100,
) -> chromadb.Collection:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/processed/chroma")
    client = _get_client(persist_dir)
    ef = _get_embedding_fn(model_name)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "source_url": c["source_url"],
                "version": c.get("version") or "",
                "section_heading": c.get("metadata", {}).get("section_heading", ""),
                "endpoint": c.get("metadata", {}).get("endpoint") or "",
                "method": c.get("metadata", {}).get("method") or "",
                "depth": c.get("depth", 0),
                "parent_chunk_id": c.get("parent_chunk_id") or "",
                "section_path": " > ".join(c.get("section_path", [])),
            }
            for c in batch
        ]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed batch %d/%d (%d chunks)", i // batch_size + 1, -(-len(chunks) // batch_size), len(batch))

    logger.info("ChromaDB index built: %d chunks in collection '%s'", collection.count(), COLLECTION_NAME)
    return collection


def load_chroma_collection(
    persist_dir: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> chromadb.Collection:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/processed/chroma")
    client = _get_client(persist_dir)
    ef = _get_embedding_fn(model_name)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def query_chroma(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 10,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(top_k, collection.count() or 1)}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    docs: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return docs

    for chunk_id, text, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - distance  # cosine distance → similarity
        docs.append({
            "chunk_id": chunk_id,
            "doc_id": meta.get("doc_id", ""),
            "text": text,
            "source_url": meta.get("source_url", ""),
            "score": round(score, 4),
            "version": meta.get("version") or None,
            "section_path": meta.get("section_path", "").split(" > ") if meta.get("section_path") else [],
            "metadata": meta,
        })

    return docs
