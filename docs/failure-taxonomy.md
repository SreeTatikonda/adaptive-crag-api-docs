# Failure Taxonomy

This document classifies the failure modes observed in the system and maps each to its corrective action.

## 1. Retrieval miss

**Definition:** The relevant documentation chunk exists in the index but is not in the top-k results for the query.

**Causes:**
- Query uses different terminology than the doc (e.g., "payment processing" vs. "PaymentIntent")
- Dense embedding collapses synonyms in a way that misses exact matches
- BM25 misses paraphrased queries

**Indicators:** Low `accepted_docs` count, all grade decisions are `reject` despite the answer existing in the corpus.

**Corrective action:** `rewrite_query` — reformulate using normalized API terminology (e.g., canonicalizing "charge a card" → "POST /v1/charges").

---

## 2. Version mismatch

**Definition:** Retrieved chunks describe the correct endpoint or behavior but for a different API version than the user queried.

**Causes:**
- Stripe docs index contains pages for multiple API versions
- Query specifies an older version but dense retrieval returns newer chunks
- Version not extracted from query

**Indicators:** `grade_results[i].version_match < 0.4`, routing decision shows low version_match average.

**Corrective action:** `alternate` — re-retrieve using ChromaDB `where` filter on the specific API version.

---

## 3. Stale documentation

**Definition:** The indexed documentation is outdated and no longer reflects the current Stripe API behavior.

**Causes:**
- Index was built without re-ingesting after a Stripe API version bump
- Changelog entries not ingested
- New endpoints added but not scraped

**Indicators:** `version_correctness` drops in longitudinal evaluation; users report answers that contradict current docs.

**Corrective action:** Re-run `make ingest` and `make index` to refresh the corpus. Use `export_metrics.py` to monitor abstention rate as a drift proxy.

---

## 4. Insufficient evidence

**Definition:** Relevant chunks are retrieved and graded as accepted, but individually none fully answers the question (each chunk is a partial piece of a multi-step answer).

**Causes:**
- Question requires combining information from multiple sections (e.g., "how do I create a subscription and handle failed payments?")
- Chunks are too small and split a conceptual unit across boundaries

**Indicators:** `accepted_docs` count ≥ 1 but `grade_results[i].sufficiency < 0.5` on all chunks.

**Corrective action:** `expand_context` — fetch parent section and adjacent chunks to reconstruct the full section context.

---

## 5. Unsupported generation

**Definition:** The generator produces a claim that is not grounded in any accepted chunk, typically by extrapolating or hallucinating details.

**Causes:**
- Generator model "fills in" common API patterns not present in the retrieved text
- Question asks about behavior at the boundary of what is documented
- Context window truncation removes relevant chunks

**Indicators:** `verify_answer` returns `verdict=abstain` with a list of `unsupported_claims`.

**Corrective action:** Verify node replaces the answer with a safe abstention message and logs the unsupported claims for human review.

---

## 6. Out-of-scope query

**Definition:** The user asks a question unrelated to Stripe API documentation (general programming, billing disputes, non-technical questions).

**Causes:**
- Misuse of the assistant
- Ambiguous phrasing that sounds technical but isn't API-related

**Indicators:** `query_type=out_of_scope` from classifier; near-zero retrieval scores across all chunks.

**Corrective action:** Route directly to abstain. The abstain message suggests the user visit stripe.com or contact Stripe support.
