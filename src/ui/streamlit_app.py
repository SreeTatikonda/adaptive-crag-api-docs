"""Streamlit chat UI with citations, trace view, and ChromaDB vector metrics dashboard."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import httpx
import streamlit as st

# Allow importing src modules when running from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Stripe API Docs — Adaptive CRAG",
    page_icon=None,
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    api_version = st.text_input("API Version filter (optional)", placeholder="e.g. 2024-06-20")
    show_trace = st.toggle("Show retrieval trace", value=True)
    show_metrics = st.toggle("Show cost & latency", value=True)

    st.divider()
    st.subheader("System Metrics")
    if st.button("Refresh metrics"):
        try:
            resp = httpx.get(f"{API_BASE}/metrics", timeout=5)
            m = resp.json()
            st.metric("Total queries", m.get("total_queries", 0))
            st.metric("Grounded rate", f"{m.get('grounded_rate', 0):.1%}")
            st.metric("Abstention rate", f"{m.get('abstention_rate', 0):.1%}")
            st.metric("Correction rate", f"{m.get('correction_rate', 0):.1%}")
            st.metric("p50 latency", f"{m.get('p50_latency_ms', 0):.0f} ms")
            st.metric("p95 latency", f"{m.get('p95_latency_ms', 0):.0f} ms")
        except Exception as exc:
            st.error(f"Could not reach API: {exc}")

# --- Tabs ---
chat_tab, vectordb_tab = st.tabs(["Chat", "Vector DB"])

# ══════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════
with chat_tab:
    st.title("Stripe API Documentation Assistant")
    st.caption("Powered by Adaptive Corrective RAG · LangGraph · Claude")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"- {c}")
            if msg["role"] == "assistant" and show_trace and msg.get("trace"):
                with st.expander("Retrieval trace"):
                    for step in msg["trace"]:
                        st.code(step, language=None)
            if msg["role"] == "assistant" and show_metrics and msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(4)
                cols[0].metric("Query type", meta.get("query_type", "—"))
                cols[1].metric("Routing", meta.get("routing_action", "—"))
                cols[2].metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms")
                cols[3].metric("Cost", f"${meta.get('token_cost_usd', 0):.4f}")

    if prompt := st.chat_input("Ask a Stripe API question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking…")

            try:
                payload = {"query": prompt, "version": api_version or None}
                resp = httpx.post(f"{API_BASE}/query", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer returned.")
                citations = data.get("citations", [])
                trace = data.get("trace", [])
                meta = {
                    "query_type": data.get("query_type", ""),
                    "routing_action": data.get("routing_action", ""),
                    "latency_ms": data.get("latency_ms", 0),
                    "token_cost_usd": data.get("token_cost_usd", 0),
                }

                placeholder.markdown(answer)

                if citations:
                    with st.expander("Citations"):
                        for c in citations:
                            st.markdown(f"- {c}")

                if show_trace and trace:
                    with st.expander("Retrieval trace"):
                        for step in trace:
                            st.code(step, language=None)

                if show_metrics:
                    cols = st.columns(4)
                    cols[0].metric("Query type", meta["query_type"])
                    cols[1].metric("Routing", meta["routing_action"])
                    cols[2].metric("Latency", f"{meta['latency_ms']:.0f} ms")
                    cols[3].metric("Cost", f"${meta['token_cost_usd']:.4f}")

                st.divider()
                fcols = st.columns([1, 1, 4])
                if fcols[0].button("Helpful"):
                    httpx.post(f"{API_BASE}/feedback", json={"query": prompt, "answer": answer, "rating": 5}, timeout=5)
                    fcols[0].success("Thanks!")
                if fcols[1].button("Not helpful"):
                    httpx.post(f"{API_BASE}/feedback", json={"query": prompt, "answer": answer, "rating": 1}, timeout=5)
                    fcols[1].info("Noted.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "trace": trace,
                    "meta": meta,
                })

            except httpx.ConnectError:
                error_msg = "Cannot connect to the API server. Run `make api` first."
                placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as exc:
                error_msg = f"Error: {exc}"
                placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ══════════════════════════════════════════════════════
# VECTOR DB TAB
# ══════════════════════════════════════════════════════
with vectordb_tab:
    st.title("ChromaDB Vector Index Dashboard")
    st.caption("Live view of the indexed Stripe API doc chunks")

    @st.cache_resource(show_spinner="Loading ChromaDB collection…")
    def _load_collection():
        from src.indexing.chroma_index import load_chroma_collection
        return load_chroma_collection()

    @st.cache_data(show_spinner="Fetching all metadata…", ttl=120)
    def _fetch_all_metadata():
        col = _load_collection()
        total = col.count()
        if total == 0:
            return [], 0
        result = col.get(include=["metadatas"])
        return result["metadatas"], total

    try:
        metadatas, total_chunks = _fetch_all_metadata()
    except Exception as exc:
        st.error(f"Could not open ChromaDB collection: {exc}")
        st.info("Run `make index` to build the index first.")
        st.stop()

    if total_chunks == 0:
        st.warning("The collection is empty. Run `make index` to populate it.")
        st.stop()

    # ── Top-level KPIs ──────────────────────────────
    unique_docs = len({m.get("doc_id", "") for m in metadatas})
    unique_versions = len({m.get("version", "") for m in metadatas if m.get("version")})
    unique_endpoints = len({m.get("endpoint", "") for m in metadatas if m.get("endpoint")})
    unique_methods = len({m.get("method", "") for m in metadatas if m.get("method")})

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total chunks", f"{total_chunks:,}")
    k2.metric("Unique docs", unique_docs)
    k3.metric("API versions", unique_versions)
    k4.metric("Endpoints", unique_endpoints)
    k5.metric("HTTP methods", unique_methods)

    st.divider()

    # ── Distribution charts ──────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Chunks by API Version")
        version_counts = Counter(m.get("version") or "unknown" for m in metadatas)
        if version_counts:
            import pandas as pd
            df_ver = pd.DataFrame(version_counts.most_common(), columns=["version", "chunks"])
            st.bar_chart(df_ver.set_index("version"))
        else:
            st.info("No version metadata found.")

    with chart_col2:
        st.subheader("Chunks by HTTP Method")
        method_counts = Counter(m.get("method") or "none" for m in metadatas)
        if method_counts:
            import pandas as pd
            df_method = pd.DataFrame(method_counts.most_common(), columns=["method", "chunks"])
            st.bar_chart(df_method.set_index("method"))
        else:
            st.info("No method metadata found.")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Chunks by Section Depth")
        depth_counts = Counter(int(m.get("depth", 0)) for m in metadatas)
        if depth_counts:
            import pandas as pd
            df_depth = pd.DataFrame(
                sorted(depth_counts.items()), columns=["depth", "chunks"]
            )
            st.bar_chart(df_depth.set_index("depth"))

    with chart_col4:
        st.subheader("Top 15 Endpoints by Chunk Count")
        ep_counts = Counter(m.get("endpoint") or "—" for m in metadatas if m.get("endpoint"))
        if ep_counts:
            import pandas as pd
            df_ep = pd.DataFrame(ep_counts.most_common(15), columns=["endpoint", "chunks"])
            st.bar_chart(df_ep.set_index("endpoint"))
        else:
            st.info("No endpoint metadata indexed.")

    st.divider()

    # ── Top docs by chunk count ──────────────────────
    st.subheader("Documents in the Index")
    doc_url_map: dict[str, str] = {}
    doc_chunk_counts: Counter = Counter()
    for m in metadatas:
        did = m.get("doc_id", "?")
        doc_chunk_counts[did] += 1
        if did not in doc_url_map:
            doc_url_map[did] = m.get("source_url", "")

    import pandas as pd
    df_docs = pd.DataFrame(
        [
            {"doc_id": did, "url": doc_url_map[did], "chunks": cnt}
            for did, cnt in doc_chunk_counts.most_common()
        ]
    )
    st.dataframe(df_docs, use_container_width=True, hide_index=True)

    st.divider()

    # ── Browse chunks with filters ───────────────────
    st.subheader("Browse Chunks")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    all_versions = sorted({m.get("version") or "" for m in metadatas if m.get("version")})
    all_endpoints = sorted({m.get("endpoint") or "" for m in metadatas if m.get("endpoint")})
    all_methods = sorted({m.get("method") or "" for m in metadatas if m.get("method")})

    sel_version = filter_col1.selectbox("Filter by version", ["(all)"] + all_versions)
    sel_endpoint = filter_col2.selectbox("Filter by endpoint", ["(all)"] + all_endpoints)
    sel_method = filter_col3.selectbox("Filter by method", ["(all)"] + all_methods)
    page_size = 10

    @st.cache_data(ttl=60)
    def _browse(version: str, endpoint: str, method: str):
        col = _load_collection()
        where_clauses = []
        if version != "(all)":
            where_clauses.append({"version": {"$eq": version}})
        if endpoint != "(all)":
            where_clauses.append({"endpoint": {"$eq": endpoint}})
        if method != "(all)":
            where_clauses.append({"method": {"$eq": method}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        kwargs = {"include": ["metadatas", "documents"]}
        if where:
            kwargs["where"] = where
        return col.get(**kwargs)

    browse_result = _browse(sel_version, sel_endpoint, sel_method)
    browse_ids = browse_result.get("ids", [])
    browse_docs = browse_result.get("documents", [])
    browse_metas = browse_result.get("metadatas", [])
    total_browse = len(browse_ids)

    st.caption(f"{total_browse} chunks match the current filters")

    if total_browse > 0:
        page_count = max(1, (total_browse + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
        start = (page - 1) * page_size
        end = min(start + page_size, total_browse)

        for i in range(start, end):
            meta = browse_metas[i] if browse_metas else {}
            cid = browse_ids[i]
            text = browse_docs[i] if browse_docs else ""
            heading = meta.get("section_heading") or meta.get("section_path") or "—"
            with st.expander(f"`{cid[:16]}…` · {heading}"):
                info_cols = st.columns(4)
                info_cols[0].write(f"**Version:** {meta.get('version') or '—'}")
                info_cols[1].write(f"**Endpoint:** {meta.get('endpoint') or '—'}")
                info_cols[2].write(f"**Method:** {meta.get('method') or '—'}")
                info_cols[3].write(f"**Depth:** {meta.get('depth', 0)}")
                st.caption(f"Source: {meta.get('source_url', '')}")
                st.text(text[:800] + ("…" if len(text) > 800 else ""))

    st.divider()

    # ── Similarity search ────────────────────────────
    st.subheader("Similarity Search")
    st.caption("Enter a query to find the closest chunks in the vector index.")

    sim_col1, sim_col2 = st.columns([4, 1])
    sim_query = sim_col1.text_input("Search query", placeholder="e.g. how do I create a payment intent?")
    top_k = sim_col2.slider("Top K", min_value=1, max_value=20, value=5)

    if sim_query:
        with st.spinner("Searching…"):
            try:
                from src.indexing.chroma_index import query_chroma
                results = query_chroma(_load_collection(), sim_query, top_k=top_k)

                if not results:
                    st.info("No results found.")
                else:
                    score_min = min(r["score"] for r in results)
                    score_max = max(r["score"] for r in results)

                    for rank, doc in enumerate(results, start=1):
                        score = doc["score"]
                        score_pct = (
                            (score - score_min) / (score_max - score_min)
                            if score_max > score_min else 1.0
                        )
                        bar = "█" * int(score_pct * 20) + "░" * (20 - int(score_pct * 20))
                        heading = (
                            " > ".join(doc["section_path"]) if doc["section_path"] else "—"
                        )
                        with st.expander(
                            f"**#{rank}** · score `{score:.4f}` `{bar}` · {heading}"
                        ):
                            rcols = st.columns(3)
                            rcols[0].write(f"**Version:** {doc['version'] or '—'}")
                            rcols[1].write(f"**Doc:** `{doc['doc_id'][:12]}…`")
                            rcols[2].write(f"**Chunk:** `{doc['chunk_id'][:16]}…`")
                            st.caption(doc["source_url"])
                            st.markdown(doc["text"][:600] + ("…" if len(doc["text"]) > 600 else ""))

            except Exception as exc:
                st.error(f"Search failed: {exc}")
