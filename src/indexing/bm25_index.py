"""BM25 lexical index using rank_bm25 for symbol-heavy technical text retrieval."""

from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.\/]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, query_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        tokens = _tokenize(query_text)
        scores = self.bm25.get_scores(tokens)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for idx, score in indexed:
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "source_url": chunk["source_url"],
                "score": round(float(score), 4),
                "version": chunk.get("version"),
                "section_path": chunk.get("section_path", []),
                "metadata": chunk.get("metadata", {}),
            })

        return results

    def save(self, path: str | None = None) -> None:
        path = path or os.getenv("BM25_INDEX_PATH", "./data/processed/bm25_index.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("BM25 index saved to %s (%d chunks)", path, len(self.chunks))

    @classmethod
    def load(cls, path: str | None = None) -> "BM25Index":
        path = path or os.getenv("BM25_INDEX_PATH", "./data/processed/bm25_index.pkl")
        with open(path, "rb") as f:
            idx = pickle.load(f)
        logger.info("BM25 index loaded from %s (%d chunks)", path, len(idx.chunks))
        return idx


def build_bm25_index(chunks: list[dict[str, Any]], save_path: str | None = None) -> BM25Index:
    idx = BM25Index(chunks)
    idx.save(save_path)
    return idx
