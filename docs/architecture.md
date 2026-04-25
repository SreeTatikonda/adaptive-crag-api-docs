# Architecture

## State machine design

The system is implemented as a `StateGraph[RAGState]` in LangGraph. Every node receives the full state and returns a partial update. Conditional edges implement the corrective routing logic.

### RAGState fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Original user query |
| `query_type` | str | Classified intent |
| `constraints` | dict | Extracted API version, endpoint, language, etc. |
| `retrieved_docs` | list | Raw retrieved chunks from hybrid search |
| `accepted_docs` | list | Chunks that passed grading |
| `rejected_docs` | list | Chunks that failed grading |
| `grade_results` | list | Per-chunk GradeResult scores |
| `routing_decision` | dict | RoutingDecision: action, reason, confidence |
| `rewritten_query` | str | Reformulated query (after rewrite node) |
| `answer` | str | Final answer text |
| `citations` | list | Cited chunk IDs and source URLs |
| `verification_result` | dict | Verifier verdict: pass/revise/abstain |
| `trace` | list | Human-readable node execution path |
| `metrics` | dict | Token counts, latency, cost, retry count |

## Node-by-node description

### 1. `classify_query`
Sends the query to Claude with a structured prompt classifying it into `fact_lookup`, `how_to`, `error_debugging`, `migration`, `sdk_usage`, or `out_of_scope`. Uses temperature=0 for deterministic classification.

### 2. `extract_constraints`
Extracts product area, API version, programming language, endpoint path, auth context, and SDK name. These are used to apply metadata filters in ChromaDB and to inform the rewrite node.

### 3. `retrieve_hybrid`
Runs dense retrieval (ChromaDB cosine similarity) and BM25 lexical search in parallel, then fuses results using Reciprocal Rank Fusion (RRF). Metadata filters from `constraints` are applied to ChromaDB to narrow by version or endpoint.

### 4. `grade_documents`
Each retrieved chunk (up to 8) is scored by Claude on:
- **Relevance** (0–1): Does it address the query?
- **Sufficiency** (0–1): Does it have enough detail?
- **Specificity** (0–1): Is it specific to the API/endpoint asked about?
- **Version match** (0–1): Does the chunk version match the query constraint?

Chunks with `decision=accept` go into `accepted_docs`.

### 5. `route_correction`
Policy engine that reads grade results and decides:
- `generate` if ≥2 accepted docs with avg relevance ≥0.6
- `expand` if relevant but insufficient accepted docs
- `rewrite` if low relevance and retries remain
- `alternate` if version mismatch detected
- `abstain` if retries exhausted or no docs

### 6. `rewrite_query`
Claude reformulates the query using missing constraints and Stripe-normalized terminology. Increments `metrics.retry_count`.

### 7. `expand_context`
Fetches parent chunks of accepted docs from ChromaDB using `parent_chunk_id` links established during chunking. Adds sibling context for structural completeness.

### 8. `rerank_documents`
Claude scores each of up to 15 chunks against the query, returning a 0–1 score. Results are re-sorted by this score before re-grading.

### 9. `generate_answer`
Claude generates an answer using only `accepted_docs`. Each claim must cite a chunk ID in `[chunk_id]` format. Temperature=0.3 for fluent but grounded output.

### 10. `verify_answer`
Claude checks whether each factual claim in the answer is supported by the retrieved context. Returns `pass`, `revise`, or `abstain`. If `abstain`, the answer is replaced with a safe abstention message.

### 11. `log_metrics`
Appends a structured JSON record to `traces.jsonl` including: timestamp, query type, routing action, retry count, node path, latency, token counts, cost, groundedness, and verification verdict.

## Corrective loop guards

To prevent infinite loops, `metrics.retry_count` is incremented on every `rewrite_query` call. The routing policy forces `abstain` when `retry_count >= routing.rewrite_max_retries` (default: 2).
