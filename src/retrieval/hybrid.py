from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from src.graph.state import RAGState

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """\
Given a developer's question and a documentation chunk, rate how useful the chunk is for answering it.
Return JSON: {"score": <0.0-1.0>, "reason": "<why>"}
1.0 = directly answers the question, 0.0 = completely off-topic.
"""


def _build_metadata_filter(constraints: dict[str, Any]) -> dict | None:
    filters: list[dict] = []

    version = constraints.get("api_version")
    if version:
        filters.append({"version": {"$eq": version}})

    endpoint = constraints.get("endpoint")
    if endpoint:
        filters.append({"endpoint": {"$eq": endpoint}})

    if len(filters) == 1:
        return filters[0]
    if len(filters) > 1:
        return {"$and": filters}
    return None


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank + 1)
        chunk_map[cid] = doc

    for rank, doc in enumerate(bm25_results):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = doc

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result: list[dict] = []
    for cid, fused_score in merged:
        doc = dict(chunk_map[cid])
        doc["score"] = round(fused_score, 6)
        result.append(doc)
    return result


def retrieve_hybrid(state: RAGState) -> dict:
    from src.indexing.chroma_index import load_chroma_collection, query_chroma
    from src.indexing.bm25_index import BM25Index

    query = state.get("rewritten_query") or state["query"]
    constraints = state.get("constraints", {})
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    top_k = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    dense_weight = float(os.getenv("DENSE_WEIGHT", "0.6"))
    bm25_weight = float(os.getenv("BM25_WEIGHT", "0.4"))

    where_filter = _build_metadata_filter(constraints)

    try:
        collection = load_chroma_collection()
        dense_results = query_chroma(collection, query, top_k=top_k, where=where_filter)
    except Exception as exc:
        logger.warning("Dense retrieval failed: %s", exc)
        dense_results = []

    try:
        bm25_idx = BM25Index.load()
        bm25_results = bm25_idx.query(query, top_k=top_k)
    except Exception as exc:
        logger.warning("BM25 retrieval failed: %s", exc)
        bm25_results = []

    merged = _reciprocal_rank_fusion(dense_results, bm25_results, dense_weight, bm25_weight)

    retrieved = [
        {
            "doc_id": d["doc_id"],
            "chunk_id": d["chunk_id"],
            "text": d["text"],
            "source_url": d["source_url"],
            "score": d["score"],
            "version": d.get("version"),
            "section_path": d.get("section_path", []),
            "metadata": d.get("metadata", {}),
        }
        for d in merged[:top_k]
    ]

    trace.append(f"retrieve_hybrid → {len(retrieved)} docs (dense={len(dense_results)}, bm25={len(bm25_results)})")
    logger.info("Retrieved %d docs after RRF fusion", len(retrieved))

    return {"retrieved_docs": retrieved, "trace": trace, "metrics": metrics}


def expand_context(state: RAGState) -> dict:
    from src.indexing.chroma_index import load_chroma_collection

    accepted = state.get("accepted_docs", [])
    retrieved = list(state.get("retrieved_docs", []))
    trace = list(state.get("trace", []))

    if not accepted:
        trace.append("expand_context → no accepted docs to expand")
        return {"retrieved_docs": retrieved, "trace": trace}

    try:
        collection = load_chroma_collection()
    except Exception as exc:
        logger.warning("Cannot load ChromaDB for expansion: %s", exc)
        return {"retrieved_docs": retrieved, "trace": trace}

    seen_ids = {d["chunk_id"] for d in retrieved}
    new_chunks: list[dict] = []

    for doc in accepted:
        parent_id = doc.get("metadata", {}).get("parent_chunk_id", "")
        if parent_id and parent_id not in seen_ids:
            try:
                result = collection.get(ids=[parent_id], include=["documents", "metadatas"])
                if result["ids"]:
                    meta = result["metadatas"][0]
                    new_chunks.append({
                        "chunk_id": parent_id,
                        "doc_id": meta.get("doc_id", ""),
                        "text": result["documents"][0],
                        "source_url": meta.get("source_url", ""),
                        "score": 0.5,
                        "version": meta.get("version") or None,
                        "section_path": meta.get("section_path", "").split(" > ") if meta.get("section_path") else [],
                        "metadata": meta,
                    })
                    seen_ids.add(parent_id)
            except Exception:
                pass

    trace.append(f"expand_context → added {len(new_chunks)} parent chunks")
    logger.info("Expanded context with %d parent chunks", len(new_chunks))
    return {"retrieved_docs": retrieved + new_chunks, "trace": trace}


def rerank_documents(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    retrieved = state.get("retrieved_docs", [])
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    if not retrieved:
        return {"retrieved_docs": retrieved, "trace": trace, "metrics": metrics}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    scored_docs: list[tuple[float, dict]] = []

    for doc in retrieved[:15]:
        prompt = f"Question: {query}\n\nChunk:\n{doc['text'][:1000]}"
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=128,
                temperature=0,
                system=_RERANK_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            parsed = json.loads(raw)
            llm_score = float(parsed.get("score", doc["score"]))
            metrics["input_tokens"] = metrics.get("input_tokens", 0) + msg.usage.input_tokens
            metrics["output_tokens"] = metrics.get("output_tokens", 0) + msg.usage.output_tokens
        except Exception:
            llm_score = doc["score"]

        doc_copy = dict(doc)
        doc_copy["score"] = round(llm_score, 4)
        scored_docs.append((llm_score, doc_copy))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    reranked = [d for _, d in scored_docs]

    trace.append(f"rerank_documents → reranked {len(reranked)} docs")
    return {"retrieved_docs": reranked, "trace": trace, "metrics": metrics}


def rewrite_query(state: RAGState) -> dict:
    query = state["query"]
    constraints = state.get("constraints", {})
    query_type = state.get("query_type", "fact_lookup")
    trace = list(state.get("trace", []))
    metrics = dict(state.get("metrics", {}))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = (
        "Rewrite this developer question about Stripe API docs to be more specific and easier to retrieve. "
        "Add endpoint names, parameter names, or version numbers if they're implied. "
        "Return only the rewritten question, nothing else."
    )
    context = f"Question type: {query_type}\nKnown constraints: {json.dumps(constraints)}\nOriginal: {query}"

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": context}],
    )

    rewritten = msg.content[0].text.strip()
    metrics["input_tokens"] = metrics.get("input_tokens", 0) + msg.usage.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + msg.usage.output_tokens

    retry_count = metrics.get("retry_count", 0) + 1
    metrics["retry_count"] = retry_count

    trace.append(f"rewrite_query (attempt {retry_count}) → {rewritten[:80]}")
    logger.info("Rewritten query: %s", rewritten)

    return {"rewritten_query": rewritten, "trace": trace, "metrics": metrics}
