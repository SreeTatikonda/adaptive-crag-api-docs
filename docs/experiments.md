# Experiment Studies

## Study 1: Chunking — flat vs. hierarchy-aware

**Hypothesis:** Hierarchy-aware chunks with section ancestry and parent-child links improve retrieval precision on nested API documentation.

**Setup:** Index the same Stripe doc corpus twice — once with flat 512-token chunks (no ancestry), once with hierarchy-aware chunks preserving `section_path` and `parent_chunk_id`. Run retrieval-only evaluation on the benchmark dataset.

**Metrics:** precision@3, recall@3, NDCG@3

**Expected finding:** Hierarchy-aware chunks improve recall on multi-hop questions (e.g., "what are the parameters for creating a subscription?") because the grader can expand to parent sections when a leaf chunk is insufficient.

---

## Study 2: Retrieval — dense vs. BM25 vs. hybrid

**Hypothesis:** Hybrid retrieval outperforms either modality alone on symbol-heavy developer queries.

**Setup:** Run three retrieval-only conditions on the benchmark: (a) dense-only ChromaDB top-10, (b) BM25-only top-10, (c) RRF-fused hybrid. Evaluate against labeled relevant doc IDs.

**Metrics:** precision@1, precision@5, recall@5, MRR

**Expected finding:** BM25 dominates for exact-match queries (endpoint names like `/v1/payment_intents`, error codes like `card_declined`). Dense dominates for semantic intent queries ("how do I handle authentication failures"). Hybrid achieves best overall.

---

## Study 3: Correction policy — none vs. static vs. adaptive

**Hypothesis:** Adaptive routing (query-type-aware + grade-signal-aware) improves groundedness and citation correctness over a fixed-threshold policy.

**Setup:** Compare three systems end-to-end: (a) no correction (retrieve→grade→always generate), (b) static CRAG (grade once, abstain if below threshold, else generate), (c) adaptive CRAG (full corrective routing with rewrite, expand, alternate, abstain).

**Metrics:** groundedness, citation correctness, abstention accuracy, correction_frequency

**Expected finding:** Adaptive routing uses correction selectively (~30–40% of queries), while static CRAG either over-abstains or under-corrects depending on threshold choice.

---

## Study 4: Version-awareness — with vs. without version metadata

**Hypothesis:** Version metadata filtering reduces wrong-version answers for queries that specify an API version.

**Setup:** Subset the benchmark to queries with explicit API version constraints. Run (a) retrieval with no version filter, (b) retrieval with ChromaDB `where` filter on version. Evaluate answer version correctness using LLM judge.

**Metrics:** version_correctness (LLM judge), precision@3 for version-constrained queries

**Expected finding:** Version filtering reduces version mismatch by surfacing chunks with matching version metadata, at the cost of slightly lower recall when the corpus has sparse version coverage.

---

## Study 5: Verification — with vs. without post-generation check

**Hypothesis:** Post-generation verification reduces unsupported claims at acceptable latency cost.

**Setup:** Run AdaptiveCRAG with and without the `verify_answer` node. Compare groundedness and unsupported claim rate. Measure additional latency introduced by verification.

**Metrics:** groundedness, unsupported_claim_rate, latency_p50, latency_p95

**Expected finding:** Verification reduces unsupported claims by ~10–20% for complex multi-hop questions, adds ~500–1500ms latency. Acceptable tradeoff for production use cases where trust is critical.

---

## Study 6: Robustness — clean vs. noisy developer queries

**Hypothesis:** The system degrades gracefully on noisy or incomplete queries rather than generating confidently wrong answers.

**Setup:** Take the benchmark questions and apply: (a) typos/misspellings, (b) truncated queries (first half only), (c) overly vague phrasing ("how do payments work?"). Compare abstention rate and groundedness vs. clean queries.

**Metrics:** abstention_accuracy, groundedness, correction_frequency

**Expected finding:** Noisy queries trigger more rewrite corrections. Vague queries trigger more abstentions. The system should never confidently answer an out-of-scope query.

---

## Study 7: Longitudinal evaluation — metric drift over time

**Hypothesis:** As Stripe updates its documentation, retrieval quality degrades without re-indexing, detectable via the evaluation harness.

**Setup:** Run the benchmark immediately after indexing. Simulate doc updates (modify a subset of chunks). Re-run benchmark without re-indexing. Compare metrics before and after simulated drift.

**Metrics:** precision@3, groundedness, version_correctness at T0 vs T1

**Expected finding:** Stale index degrades version_correctness and groundedness on updated endpoints. The observability logger's correction_frequency and abstention_rate serve as early indicators of index staleness.
