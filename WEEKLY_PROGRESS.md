# Weekly Progress — Mini HR Policy RAG App

**Project:** Ask-My-HR-Documents (Track C — HR Policy)
**Stack:** Python · sentence-transformers (BGE-small) · Qdrant · OpenRouter
**Documents:** `employee_handbook.md` + `policy_addendum.md`

---

## Week 3 — Retrieval & RAG (Optimization)

### Goal
The base app was working but had known weaknesses. Week 3 was about making it more accurate and production-ready by fixing four specific issues.

### What We Did

#### 1. Raised top_k from 4 to 8
- **Problem:** For compound questions (e.g. "Can a new hire work remotely?"), the retriever missed the probation-period chunk entirely because only 4 results were fetched and the relevant chunk ranked 5th or lower.
- **Fix:** Changed the default retrieval count from 4 to 8 across `query.py` and `eval.py`.
- **Files changed:** `query.py`, `eval.py`

#### 2. Added `requirements.txt`
- **Problem:** No pinned dependencies — impossible to reproduce the environment on another machine.
- **Fix:** Created `requirements.txt` with exact versions for all direct dependencies (`sentence-transformers`, `qdrant-client`, `requests`, `pypdf`).
- **Files changed:** `requirements.txt` (new)

#### 3. Parent-Child Chunking
- **Problem:** The app had to choose between two bad options — small chunks (precise retrieval, poor generation context) or large chunks (good generation context, diluted retrieval). FINDINGS.md showed both had real failure cases.
- **Fix:** A new chunking strategy that stores small `###` subsection chunks for embedding (precise retrieval) but keeps the full `##` parent section text in the payload (sent to LLM for richer context). No more tradeoff.
- **New collection:** `hr_policy_parent_child`
- **Files changed:** `ingest.py`, `vectorstore.py`

#### 4. Cross-Encoder Re-ranking
- **Problem:** Dense vector retrieval ranks by approximate similarity, not by true relevance to the exact question.
- **Fix:** After fetching `top_k × 3` candidates via dense search, a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores every candidate against the actual question and returns the best `top_k`. Much more precise than vector similarity alone.
- **New flag:** `--no-rerank` to disable when speed matters
- **Files changed:** `query.py`

### Week 3 Result
Three collections available: `hr_policy_subsection`, `hr_policy_section`, `hr_policy_parent_child`. Default query uses parent-child with cross-encoder re-ranking.

---

## Week 4 — Debugging Retrieval (Hybrid Search + Failure Analysis)

### Goal
"Wrong sometimes" is useless to fix. Week 4 was about finding out exactly *why* the app gets things wrong, fixing it with one measurable change, and proving the fix with a before/after number.

### Core Concept: Two Types of Failure

| Failure Type | What It Means | Fix |
|---|---|---|
| **Retrieval Failure** | The right document never appeared in top-k results | Fix the search |
| **Generation Failure** | Right document was retrieved, but LLM answered badly | Fix the prompt or model |

These need completely different fixes. A smarter model does nothing if retrieval is broken.

### What We Did

#### 1. Failure Classification
- **What:** Every evaluation result is now automatically labelled — `RETRIEVAL FAILURE`, `GENERATION FAILURE`, or `OK`.
- **Why:** Without knowing which type of failure it is, any fix is a guess.
- **How:** For each question, checks if the expected source document appeared in top-3. If not → retrieval failure. If yes but answer wrong → generation failure.
- **Files changed:** `eval.py`

#### 2. hit-rate@3 Metric
- **What:** For each question, measures whether the correct source document appeared in the top 3 retrieved chunks. Reports a percentage score overall.
- **Why:** Gut feeling is not proof. A before/after number is proof.
- **Alternative considered:** Eyeballing answers manually — not repeatable, not scalable.
- **Why hit-rate@3 specifically:** Standard retrieval metric. Fast to compute, requires no LLM call, directly measures retrieval quality.
- **Files changed:** `eval.py`

#### 3. Hybrid Search — BM25 + Dense + RRF (the "one change")
- **What:** Two search methods run together — BM25 keyword search and dense vector search — combined into one ranked list using Reciprocal Rank Fusion (RRF).
- **Why:** Dense search finds documents by meaning but misses exact keyword matches (policy codes, section names, specific terms). BM25 catches those. Together they cover both.
- **Example:** "Addendum 4 remote work policy" — dense search struggles with the label "Addendum 4". BM25 finds it instantly by exact keyword.
- **RRF formula:** Each retriever contributes `1 / (60 + rank)` to a combined score. Top results from both retrievers float to the top.
- **Alternative considered:** Dense only (misses keywords) · BM25 only (misses meaning) · Cohere Rerank API (paid, external dependency).
- **Why this specifically:** No new infrastructure. BM25 runs in memory. Standard approach for production RAG systems. Directly addresses retrieval failures on keyword-heavy queries.
- **New library:** `rank_bm25`
- **Files changed:** `hybrid.py` (new), `vectorstore.py`, `query.py`, `requirements.txt`

#### 4. Before/After Compare Table
- **What:** `eval.py` now runs a side-by-side comparison — dense-only retrieval vs hybrid retrieval — for every question, and prints a table with HIT/MISS per question and overall hit-rate@3 percentage.
- **Why:** Proves the hybrid change helped (or didn't). No LLM calls needed — pure retrieval measurement.
- **Files changed:** `eval.py`

#### 5. Inspection View (`debug.py`)
- **What:** New script that shows the full pipeline for any question — retrieved chunks with source, score, text preview (expected source marked with `*`), final answer, and failure diagnosis.
- **Why:** Without this, you're blind to what the retriever actually fetched. This makes the invisible visible.
- **Usage:** Run for all eval questions, or pass a single question with `--question`.
- **Alternative considered:** Print statements scattered in code — messy and not reusable.
- **Files changed:** `debug.py` (new)

#### 6. Retry with Backoff on Rate Limits
- **What:** `call_openrouter()` now retries up to 4 times with increasing wait times (5s → 10s → 20s → 40s) when it hits a 429 rate limit, instead of crashing.
- **Why:** Free-tier OpenRouter models are shared across all users. Eval runs fire many back-to-back LLM calls and hit rate limits without this.
- **Files changed:** `query.py`

### Week 4 Result
The app can now explain why it fails, not just that it fails. Hybrid search improves retrieval on keyword queries. hit-rate@3 gives a before/after number to prove any change works.

---

## File Summary

| File | Week 3 | Week 4 |
|---|---|---|
| `query.py` | top_k=8, cross-encoder re-ranking, parent_text in prompt | Hybrid flag, retry backoff, model switch |
| `ingest.py` | Parent-child chunking added | — |
| `vectorstore.py` | payload spread (all fields returned) | `get_all()` for BM25 index, `id` in hits |
| `eval.py` | top_k=8 | Failure labels, hit-rate@3, before/after compare |
| `hybrid.py` | — | New — BM25 + dense + RRF |
| `debug.py` | — | New — inspection view |
| `requirements.txt` | Created with 4 packages | Added `rank_bm25` |
| `list_models.py` | — | New — lists available free OpenRouter models |

---

## Collections in Qdrant

| Collection | Chunk Strategy | Best For |
|---|---|---|
| `hr_policy_subsection` | One chunk per `###` heading | Precise single-fact retrieval |
| `hr_policy_section` | One chunk per `##` heading (all subsections merged) | Broad context queries |
| `hr_policy_parent_child` | Embed `###` child, return `##` parent to LLM | Best of both — **default** |

---

## How to Run

```bash
# Start Qdrant
docker start qdrant-hr

# Set API key (cmd)
set OPENROUTER_API_KEY=your-key-here

# Navigate to project
cd D:\AI\rag-hr-app

# Ingest (only if data changed)
venv\Scripts\python.exe ingest.py

# Ask questions
venv\Scripts\python.exe query.py --collection hr_policy_parent_child

# Inspect failures
venv\Scripts\python.exe debug.py

# Run full evaluation with before/after comparison
venv\Scripts\python.exe eval.py
```
