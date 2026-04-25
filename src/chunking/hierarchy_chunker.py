"""Split documents into chunks that remember which section they came from."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = _TOKENIZER.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_TOKENIZER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _make_chunk_id(doc_id: str, section_path: list[str], idx: int) -> str:
    key = f"{doc_id}|{'|'.join(section_path)}|{idx}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def chunk_document(
    doc: dict[str, Any],
    chunk_size: int = 512,
    overlap: int = 64,
    min_tokens: int = 50,
) -> list[dict[str, Any]]:
    doc_id = doc["doc_id"]
    version = doc.get("version")
    source_url = doc.get("url", "")
    base_metadata = doc.get("metadata", {})
    base_metadata["title"] = doc.get("title", "")
    base_metadata["endpoint"] = doc.get("endpoint")
    base_metadata["method"] = doc.get("method")
    base_metadata["sdk"] = doc.get("sdk", [])

    chunks: list[dict[str, Any]] = []
    sections = doc.get("sections") or [{"section_path": doc.get("section_path", []), "content": doc.get("content", "")}]

    for section in sections:
        section_path: list[str] = section.get("section_path", [])
        content: str = section.get("content", "").strip()
        if not content:
            continue

        depth = len(section_path)
        sub_chunks = _chunk_text(content, chunk_size, overlap)

        parent_chunk_id: str | None = None
        for idx, sub_text in enumerate(sub_chunks):
            if _count_tokens(sub_text) < min_tokens:
                continue

            chunk_id = _make_chunk_id(doc_id, section_path, idx)

            chunk: dict[str, Any] = {
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_chunk_id,
                "doc_id": doc_id,
                "section_path": section_path,
                "depth": depth,
                "version": version,
                "text": sub_text,
                "source_url": source_url,
                "metadata": {**base_metadata, "section_heading": section_path[-1] if section_path else ""},
            }
            chunks.append(chunk)
            if idx == 0:
                parent_chunk_id = chunk_id

    return chunks


def chunk_corpus(
    docs: list[dict[str, Any]],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)
    logger.info("Created %d chunks from %d documents", len(all_chunks), len(docs))
    return all_chunks
