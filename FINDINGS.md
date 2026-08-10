# Findings — Mini "Ask My HR Documents" RAG App

**Track:** C — HR Policy
**Stack:** Python, local/free — `sentence-transformers` (BGE-small) for embeddings, Chroma for vector storage, Ollama for generation
**Documents:** `employee_handbook.md` (base handbook, v3.0) + `policy_addendum.md` (new addendum, several sections superseding/extending the base handbook)

## 1. Setup

- **Embedding model:** `BAAI/bge-small-en-v1.5` (local, via `sentence-transformers`)
- **Vector store:** Chroma (persistent, local)
- **Generation model:** Ollama, tested with two sizes — `llama3.2:3b` and `llama3.1:8b`
- **Chunking:** structure-aware, split along the documents' own Markdown headings rather than a fixed character count, in two granularities to compare directly:
  - `hr_policy_subsection` — one chunk per `###` sub-heading (25 chunks, avg 278 chars)
  - `hr_policy_section` — one chunk per `##` heading, merging all its sub-headings (13 chunks, avg 539 chars)

## 2. Chunk-size experiment (the required comparison)

Ran the same question set against both collections. Basic single-fact questions (sick leave days, probation length, bonus cap) and the "not in the documents" refusal test passed identically on both — chunk size didn't matter there.

The difference showed up on questions where the correct fact was **one specific sentence embedded inside a larger passage**:

| Question | `hr_policy_section` (539 chars avg) | `hr_policy_subsection` (278 chars avg) |
|---|---|---|
| "Can employees paste customer data into an unapproved AI tool?" | Retrieved the right chunk, but answered "I don't know" — failed to extract the fact from a denser chunk | Correctly answered "no," citing Addendum 4 |
| "Can a secondary caregiver split parental leave into two blocks?" | Retrieved the right chunk (distance 0.46), still answered "I don't know" | Correctly answered "yes, within 18 months," citing Addendum 1 and noting it supersedes the handbook |

**Conclusion:** for a small local LLM, smaller, single-fact chunks were measurably more reliable than larger merged-section chunks, even when retrieval pulled the correct chunk in both cases. Larger chunks diluted a specific fact enough that generation failed to extract it. This is the semantic-dilution tradeoff — smaller chunks cost some surrounding context but gained extraction reliability.

## 3. Grounded generation & refusal behavior

- Every in-scope answer correctly cited `(Source: <file>, Section: <heading>)`.
- Out-of-scope questions ("stock options policy," "office pet policy," "stock buybacks") correctly triggered *"I don't know based on the provided documents"* — no hallucination in any of these tests.
- Noticed a usable signal: correct answers had a top retrieval distance around **0.2–0.5**; genuinely unanswerable questions had a best distance of **0.65+**. A distance threshold could be added as a cheap pre-filter before even calling the LLM.

## 4. Failure taxonomy — a multi-hop question exposed four distinct, separable failure modes

Test question: *"A new employee is 45 days into their job and wants to start working remotely 2 days a week. Is that allowed right now, and if not, when would it become allowed?"* — this requires combining two facts from two different sections (90-day probation rule + remote-work eligibility rule), so it's a genuine multi-hop test, not a lookup.

| # | Failure mode | Symptom | What fixed it |
|---|---|---|---|
| 1 | **Retrieval failure** | At `top_k=4`, the probation-period section never entered the top results at all — it's semantically distant from "remote work" | Raised `top_k` to 8 |
| 2 | **Extraction failure** | Correct chunk retrieved, but a large merged chunk diluted the specific fact enough that the model missed it (same pattern as §2) | Switched to smaller `hr_policy_subsection` chunks |
| 3 | **Reasoning/composition failure** | Correct chunk retrieved and even correctly quoted, but the model never compared 45 vs. 90 — restated the policy without applying it to the specific case, and answered "I don't know" despite having everything it needed | Added a chain-of-thought instruction (explicit "Reasoning:" step forcing numeric comparison) — **improved fact extraction but did not fully fix synthesis** on `llama3.2:3b` |
| 4 | **Model capability ceiling** | Even with the CoT prompt, the 3B model's Reasoning section correctly concluded "45 < 90, not eligible yet" — then the Answer section still contradicted it, defaulting to "I don't know" | Swapped to `llama3.1:8b` — Reasoning became fully correct, but the same Answer/Reasoning contradiction still occurred |
| 5 | **Prompt-design failure** | Even the 8B model's correct internal reasoning got discarded because the prompt's blanket rule ("if the context doesn't contain enough info, say I don't know") triggered on the harder second half of the compound question, wiping out the correctly-solved first half | Rewrote the prompt to require independent per-part answers and explicit Answer/Reasoning consistency — **this finally resolved the contradiction**: model answered "not allowed now... eligible once probation completes" with both sources cited |

**Residual known gap (documented, not fixed):** even after the fix, the model conflated the *addendum's calendar effective date* (2025-07-01) with the *employee's personal 90-day threshold date* when both numbers were present in context — a subtler date-disambiguation issue that model-size or chunking changes alone weren't shown to fix.

## 5. Bug caught during build (not a model issue — a pipeline bug)

The first version of the "subsection" chunker split only on `###` headings. The addendum document used only `##` headings with no `###` sub-headings, so it silently produced **zero chunks for the entire addendum** in that collection — meaning all the "new policy" content was invisibly missing from search results. Caught by logging per-file chunk counts during ingestion (`employee_handbook.md: 20 chunks / policy_addendum.md: 0 chunks` — the zero was the tell). Fixed by falling back to `##`-splitting for any document with no `###` headings.

**Takeaway:** always log chunk counts per source file at ingestion time — a chunking bug that drops an entire document produces no error, just silently worse answers.

## 6. Summary for the mentor checklist

- ✅ App answers correctly from the documents, with citations on every answer.
- ✅ Refuses ("I don't know") on questions genuinely unanswerable from the documents, across all models/configs tested.
- ✅ Tried two chunk sizes and observed a concrete difference: smaller chunks won on fact-extraction reliability for small local models.
- ✅ Went further than required: isolated *why* larger chunks failed (dilution) vs. *why* even correct retrieval could still fail (reasoning ceiling and prompt design), across two model sizes.
