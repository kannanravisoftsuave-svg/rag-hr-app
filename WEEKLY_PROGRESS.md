# Weekly Progress — Mini HR Policy RAG App

**Project:** Ask-My-HR-Documents (Track C — HR Policy)
**Program:** 6-week AI Engineering curriculum — Module 1 (Foundations, Prompting) →
Module 2 (Retrieval & RAG, Debugging Retrieval) → Module 3 (Error Analysis, Evals)
**Stack:** Python · sentence-transformers (BGE-small) · Qdrant · OpenRouter · FastAPI · React · Langfuse

This file tracks what was learned and built each week, in curriculum order (Week 1 → Week 6),
plus a full "how to run this on a different machine" guide at the bottom. Keep this file
up to date — it's the single place a new contributor (or future you) can read to understand
both the reasoning behind the app and how to get it running from zero.

---

## Week 1 — Foundations (How a Language Model Actually Works)

**Format:** theory only, no coding this week (by design).

### What We Learned
- A language model is a program that predicts the next token, over and over — it doesn't
  "look up" facts, so **sounding right and being right are two different things to it**.
  This is *why* hallucination happens, not just a name for it.
- **Tokens** are the billing unit — a token is roughly a word-piece, and every prompt +
  context + answer costs tokens. Understanding tokens is understanding the bill.
- **Temperature** controls randomness in the output — low temperature (~0) gives consistent,
  repeatable answers; high temperature gives more varied/creative ones.
- **Embeddings** turn text into numbers (vectors) so a computer can compare *meaning*, not
  just exact words — this is the foundation everything in Week 3 (RAG) depends on.
- Model families (GPT, Claude, LLaMA), decoder-only vs encoder-only architectures, and roughly
  what different models cost, so a model can be picked deliberately, not by default.

### What We Did
No app code this week (per the brief). The concepts show up later in the actual app:
- `temperature=0.1` in `query.py` — a deliberate choice for a factual HR bot, not a default
- `FALLBACK_MODELS` chain in `query.py` — picks a different model if the primary one is rate-limited
- **Known gap (not yet closed):** token usage / cost per query is not logged anywhere, even
  though `tracing.py` (Langfuse) already has the infrastructure to carry it. This is the
  most direct unfinished thread back to Week 1's "tokens = your bill" lesson.

---

## Week 2 — Prompting, Structured Output & Tool Calling

**Format:** bridge week — first code written this week.

### What We Learned
- A good prompt is written carefully with examples, not typed off the cuff — prompt anatomy,
  zero-shot vs few-shot, and chain-of-thought (CoT) reasoning.
- Free-text answers can't be trusted by code. **Structured output** (JSON, validated against a
  schema — e.g. with Pydantic or the `instructor` library) is what lets a program safely use an
  LLM's answer instead of just displaying it.
- **Validation & retry** — check the answer is valid, and ask again (politely, with the specific
  error) when it isn't, instead of shipping something broken.
- **Tool calling** lets the AI *ask* your code to do something (a lookup, a calculation) — but
  your code runs it, not the AI. The AI never gets to execute anything directly.
- Simple guardrails: catching bad input and refusing safely instead of inventing an answer.

### What We Did
- The original app (`query.py`) used a **free-text prompt** — the model is asked to write
  `Quotes:` / `Reasoning:` / `Answer:` sections as prose, and nothing validated it actually
  complied. This shipped working answers, but the exact Reasoning/Answer-contradiction bug
  found in Week 5 (`FINDINGS.md` #4) is a direct consequence of this: a text-parsing pipeline
  has no way to *catch* a model contradicting itself.
- **Retrofitted in a later session** — `structured_answer.py`:
  - A Pydantic `HRAnswer` schema: `quotes`, `reasoning`, `answer`, `sources`, `is_refusal`
  - Calls OpenRouter in JSON mode (`response_format: {"type": "json_object"}`)
  - On invalid/malformed JSON, retries with the specific validation error fed back into the
    prompt — this is the "validation & retry" loop the brief asked for
  - **Additive, not a replacement** — the original `query.ask()` path still works unchanged.
    Opt into the new path with `python query.py --structured` on the CLI, or
    `"use_structured_output": true` in a `POST /query` request body.
- **Tool calling was not implemented.** This app is pure retrieve-then-generate — there's no
  real action (a lookup, a booking, a calculation) for the model to trigger. Recorded here as a
  deliberate scope decision, not a missed requirement.

---

## Week 3 — Retrieval & RAG (Build + Optimization)

**Format:** build week.

### What We Learned
- Why RAG exists: a model only knows public training data — ask it about your own documents
  and it will confidently invent an answer. RAG = "let the AI read the right documents first,
  then answer from them."
- **Chunking strategies** and why chunk size matters — too large dilutes a specific fact inside
  a big passage (hurts extraction); too small loses surrounding context.
- **Embeddings & dense retrieval** — bi-encoder (fast, used for search) vs cross-encoder
  (slower, more accurate, used for reranking) and how they trade off.
- Vector databases (HNSW-based — Qdrant/Chroma/pgvector), similarity search & top-K,
  metadata filtering.
- **Grounded generation & citations** — make the model quote its source, and say "I don't know"
  instead of guessing when the answer isn't in the documents.

### What We Did

The base app was working but had known weaknesses. This week made it more accurate and
production-ready by fixing four specific issues:

#### 1. Raised `top_k` from 4 to 8
- **Problem:** for compound questions (e.g. "Can a new hire work remotely?"), the retriever
  missed the probation-period chunk entirely — with only 4 results fetched, the relevant chunk
  ranked 5th or lower and never made it in.
- **Fix:** changed the default retrieval count from 4 to 8 in `query.py` and `eval.py`.

#### 2. Added `requirements.txt`
- **Problem:** no pinned dependencies — impossible to reproduce the environment on another machine.
- **Fix:** created `requirements.txt` with exact versions for all direct dependencies.

#### 3. Parent-child chunking
- **Problem:** the app had to choose between two bad options — small chunks (precise retrieval,
  poor generation context) or large chunks (good generation context, diluted retrieval).
  `FINDINGS.md` shows real failure cases on both sides.
- **Fix:** a new chunking strategy stores small `###` subsection chunks for embedding (precise
  retrieval) but keeps the full `##` parent section text in the payload (sent to the LLM for
  richer context). No more tradeoff.
- **New collection:** `hr_policy_parent_child` (see the collection table below)

#### 4. Cross-encoder reranking
- **Problem:** dense vector retrieval ranks by approximate similarity, not true relevance to
  the exact question.
- **Fix:** after fetching `top_k × 3` candidates via dense search, a cross-encoder model
  (`ms-marco-MiniLM-L-6-v2`) re-scores every candidate against the actual question and returns
  the best `top_k`. Much more precise than vector similarity alone.
- **New flag:** `--no-rerank` to disable when speed matters more than precision.

**Result:** three collections available (`hr_policy_subsection`, `hr_policy_section`,
`hr_policy_parent_child`). Default query uses parent-child with cross-encoder reranking.

---

## Week 4 — Debugging Retrieval (Hybrid, Reranking & Failure Separation)

**Format:** build week.

### What We Learned
- **The two kinds of failure**, and why they need completely different fixes:

  | Failure Type | What It Means | Fix |
  |---|---|---|
  | **Retrieval failure** | The right document never appeared in top-k results | Fix the search |
  | **Generation failure** | Right document retrieved, but the LLM answered badly | Fix the prompt or model |

  A smarter model does nothing if the problem is retrieval — you'd pay more and fix nothing.
- **Hybrid search** — combining meaning-based (dense) and exact-word (BM25/keyword) search
  catches both "what does the policy say" and exact codes/names that pure semantic search misses.
- **Reranking** — a second pass (cross-encoder) that pushes the best result to the top.
- Measure, don't eyeball: `hit-rate@k`, `recall@k`, `MRR` — a before/after number, not a gut feeling.
- (Studied but not required/implemented this project: query rewriting, HyDE, MMR — noted as
  topics covered, not necessarily built.)

### What We Did

"Wrong sometimes" is useless to fix. This week was about finding out exactly *why* the app
gets things wrong, fixing it with one measurable change, and proving the fix with a
before/after number.

#### 1. Failure classification
- Every eval result is automatically labeled `RETRIEVAL FAILURE`, `GENERATION FAILURE`, or `OK`
  by checking whether the expected source document appeared in the top-3 results — if not,
  retrieval failure; if yes but the answer's wrong, generation failure.
- **Files:** `eval.py`

#### 2. hit-rate@3 metric
- For each question, measures whether the correct source document appeared in the top 3
  retrieved chunks, reported as a percentage.
- **Why hit-rate@3 specifically:** standard retrieval metric, fast (no LLM call needed),
  directly measures retrieval quality. (Alternative considered and rejected: eyeballing answers
  manually — not repeatable, not scalable.)
- **Files:** `eval.py`

#### 3. Hybrid search — BM25 + dense + RRF (the "one change")
- Two search methods run together — BM25 keyword search and dense vector search — combined
  into one ranked list via Reciprocal Rank Fusion (RRF).
- **Why:** dense search finds documents by meaning but misses exact keyword matches (policy
  codes, section names). BM25 catches those. Together they cover both.
- **Example:** "Addendum 4 remote work policy" — dense search struggles with the label
  "Addendum 4"; BM25 finds it instantly by exact keyword match.
- **RRF formula:** each retriever contributes `1 / (60 + rank)` to a combined score; top
  results from both retrievers float to the top.
- **Alternatives considered:** dense-only (misses keywords), BM25-only (misses meaning),
  Cohere Rerank API (paid, adds an external dependency) — hybrid BM25+dense needs no new
  infrastructure and runs in memory.
- **New library:** `rank_bm25`. **Files:** `hybrid.py` (new), `vectorstore.py`, `query.py`

#### 4. Before/after compare table
- `eval.py` runs a side-by-side comparison — dense-only vs hybrid retrieval — for every
  question, printing HIT/MISS per question plus an overall hit-rate@3 percentage. No LLM calls
  needed — pure retrieval measurement.

#### 5. Inspection view (`debug.py`)
- Shows the full pipeline for any question — retrieved chunks with source/score/text preview
  (expected source marked with `*`), the final answer, and a failure diagnosis, side by side.
  Without this, you're blind to what the retriever actually fetched.

#### 6. Retry with backoff on rate limits
- `call_openrouter()` retries up to 4 times with increasing wait (5s → 10s → 20s → 40s) on a
  429 rate limit instead of crashing — needed because eval runs fire many back-to-back LLM
  calls against shared free-tier models.

**Result:** the app can now explain *why* it fails, not just that it fails. Hybrid search
measurably improves retrieval on keyword-heavy queries, and hit-rate@3 gives a repeatable
before/after number for any future change.

---

## Week 5 — Error Analysis (Reading Traces Like a Professional)

**Format:** build week.

### What We Learned
- Reading your own app's failures by hand is the one part that can't be automated — it's how
  you discover problems you never guessed were there.
- A **trace** is a full record of one request (question, what was fetched, what was answered)
  — complete enough to replay later.
- Take a **fair, random sample**, not cherry-picked "nice" examples, when reading failures.
- Write one honest sentence about what went wrong in each failure — *before* deciding on
  categories (open coding), so the categories come from the data, not a preconception.
- Rank problem types by frequency × severity — you can't fix everything at once, ranking tells
  you what to fix first. Then pick one target and write down what you expect to happen.

### What We Did
- `FINDINGS.md` — read ~20 real answers from the app, wrote an honest note on each failure,
  grouped the notes into named, ranked problem categories.
- One multi-hop test question ("a new employee 45 days in, wants remote work — allowed?")
  exposed **five separable failure modes**, each isolated and fixed at a different point in the
  pipeline:

  | # | Failure mode | Fixed by |
  |---|---|---|
  | 1 | Retrieval failure (right chunk never in top-k) | Raised `top_k` to 8 |
  | 2 | Extraction failure (chunk too diluted to extract the fact) | Smaller subsection chunks |
  | 3 | Reasoning/composition failure (never applied the fact to the case) | Chain-of-thought prompt instruction |
  | 4 | Model capability ceiling (3B model's Answer contradicted its own correct Reasoning) | Swapped to an 8B model |
  | 5 | Prompt-design failure (a blanket "say I don't know" rule wiped out a correctly-solved part) | Rewrote the prompt to require independent per-part answers |

- **One residual gap documented, not hidden:** even after all five fixes, the model can still
  conflate the addendum's calendar effective date with an employee's personal tenure-based
  threshold date when both appear together — chunking/model-size changes alone didn't fix it.
  This is tracked as a permanent regression test in Week 6 (`known_issue=True` in `eval.py`),
  not swept under the rug.
- `questions.md` — categorized sample question bank (basic facts, addendum-supersedes-handbook,
  addendum-only content, should-refuse, multi-hop, known-shaky edge cases) for manual testing
  and demos.

---

## Week 6 — Evals (Measuring Whether a Change Actually Helped)

**Format:** build week.

### What We Learned
- Every prompt/retrieval change might fix one thing and break another — without automatic
  tests, you're guessing; with them, you get a number.
- Build a **test set**: real questions with a way to score the answer, that runs with one command.
- Turn last week's real failures into **permanent regression tests**, so they can't come back
  unnoticed.
- Do the free, simple checks first — rule-based assertions (is a source cited? did it refuse
  when it should?) — before reaching for an LLM judge.
- **LLM-as-judge** for the harder stuff a rule can't check (tone, multi-part reasoning) — but
  **never trust an unvalidated judge**. Check it agrees with your own grading on a sample first.
- Measure a change with a before-and-after score, broken down **per problem type**, not one
  blurry overall number.

### What We Did
- `eval.py` — one-command test set: keyword-match scoring, retrieval hit-rate@3, failure
  classification, results grouped by problem-type tag.
- **Regression tests** — the real Week 5 failures (the multi-hop reasoning bug, and the
  documented-unresolved date-disambiguation gap) turned into permanent `QUESTIONS` entries, so
  a future change can't silently regress them.
- `judge.py` — a hand-rolled LLM-as-judge (OpenRouter + a grading rubric) for questions a
  keyword match can't score well (e.g. did the model's Answer actually agree with its own
  Reasoning, not just contain the right words).
- `judge_calibration.py` — validates `judge.py` against your own grading before trusting it:
  generates real answers, you grade them 1-5 by hand, then the script reports how often the
  judge agrees with you (target: ≥80% within 1 point).
- `ragas_eval.py` — the same question set, scored instead with **RAGAS**'s pre-built,
  pre-tested metrics (`faithfulness`, `answer_relevancy`, `context_precision`,
  `context_recall`) — an alternative to writing and validating your own judge rubric.
- `compare_eval_runs.py` — diffs two saved `eval.py --save` runs, per problem type, to prove a
  change actually helped (not just "felt better").
- `EVALS_SETUP.md` — install/run guide for all of the above, including the extra
  `pip install` needed only for `ragas_eval.py`.

---

## File Summary

| File | Week | What it does |
|---|---|---|
| `query.py` | 1, 2, 3, 4 | CLI entrypoint; `temperature`/fallback models (W1); `--structured` flag (W2); top_k=8, cross-encoder rerank (W3); hybrid flag, retry backoff (W4) |
| `structured_answer.py` | 2 | Pydantic `HRAnswer` schema + JSON-mode LLM calls + validation/retry |
| `ingest.py` / `ingestion/` | 3 | Parent-child chunking, document loading |
| `vectorstore.py` | 3, 4 | Qdrant wrapper; `get_all()` for BM25 index (W4) |
| `hybrid.py` | 4 | BM25 + dense + RRF fusion |
| `debug.py` | 4 | Inspection view — chunks + answer + diagnosis side by side |
| `FINDINGS.md` | 5 | Error analysis write-up — 5-layer failure taxonomy, ranked |
| `questions.md` | 5 | Categorized sample question bank |
| `eval.py` | 4, 6 | hit-rate@3, failure labels (W4); regression tests, tag grouping, `--judge`, `--save` (W6) |
| `judge.py` | 6 | Hand-rolled LLM-as-judge |
| `judge_calibration.py` | 6 | Validates the judge against human grading |
| `ragas_eval.py` | 6 | RAGAS-based alternative eval |
| `compare_eval_runs.py` | 6 | Before/after diff, per problem type |
| `requirements.txt` | 3, 4, 6 | Pinned dependencies (created W3, `rank_bm25` added W4, RAGAS deps added W6) |
| `list_models.py` | 4 | Lists available free OpenRouter models |

---

## Collections in Qdrant

| Collection | Chunk Strategy | Best For |
|---|---|---|
| `hr_policy_subsection` | One chunk per `###` heading | Precise single-fact retrieval |
| `hr_policy_section` | One chunk per `##` heading (all subsections merged) | Broad context queries |
| `hr_policy_parent_child` | Embed `###` child, return `##` parent to LLM | Best of both — **default** |

---

## How to Run This Project — Full Guide for a New Machine

This is the complete path from a fresh clone to a working app, in order. For more detail on
any one step, see `SETUP.md` (base app) and `EVALS_SETUP.md` (Week 6 eval scripts) — this
section is the condensed, one-stop version for quick reference.

### 0. Prerequisites

| Tool | Why |
|---|---|
| Python 3.10+ | Runs the app and all scripts |
| Docker Desktop | Runs Qdrant (the vector database) |
| Node.js 18+ | Only needed if you want the React UI |
| An OpenRouter API key (free) | https://openrouter.ai/keys — powers the LLM calls |
| A Langfuse account (free, optional) | https://cloud.langfuse.com — for tracing/observability |

### 1. Clone and set up Python

```bash
git clone https://github.com/kannanravisoftsuave-svg/rag-hr-app.git
cd rag-hr-app
python -m venv venv
```

Activate it:
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

Install the base dependencies:
```bash
pip install -r requirements.txt
```
This installs everything needed for Weeks 1-5 and the `eval.py`/`judge.py` scripts from Week 6.
It does **not** yet install `ragas`/`langchain-openai`/`langchain-huggingface` — those are in
`requirements.txt` too, so the same command above gets them as well. If `ragas_eval.py`
specifically fails to import, re-run `pip install -r requirements.txt` — those three packages
are the newest additions and the most likely to need a fresh install.

### 2. Set your environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:
```env
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
QDRANT_URL=http://localhost:6333
```
(Langfuse keys are optional — the app runs fine without them, you just won't get traces.)

### 3. Start Qdrant (the vector database)

First time only:
```bash
docker run -d --name qdrant-hr -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```
Every time after that, just:
```bash
docker start qdrant-hr
```
Check it's up: open http://localhost:6333/dashboard in a browser.

### 4. Load the documents (ingest)

```bash
python ingest.py
```
This reads `data/employee_handbook.md` and `data/policy_addendum.md`, chunks them (three
different ways — see the Collections table above), embeds each chunk, and stores them in
Qdrant. Only needs to be re-run if the documents change.

### 5. Run the app

**Option A — CLI only (fastest way to ask a question):**
```bash
python query.py --collection hr_policy_parent_child
```
Type a question, press Enter. Add `--structured` to use the Week 2 Pydantic-validated path
instead of free-text parsing.

**Option B — Full app (FastAPI + React UI):**
```bash
# Terminal 1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

# Terminal 2
cd ui-react
npm install
npm run dev
```
Open http://localhost:3000.

**Option C — Streamlit (legacy, simpler):**
```bash
streamlit run ui/app.py
```
Opens at http://localhost:8501.

### 6. Debug a specific failure

```bash
python debug.py                        # every question in the eval set
python debug.py --question "..."       # one specific question
```
Shows retrieved chunks (with the expected source marked `*`), the final answer, and a
retrieval-vs-generation diagnosis.

### 7. Run the evals (Week 6)

```bash
python eval.py                          # rule-based scoring + hit-rate + regression tests
python eval.py --judge                  # also run the hand-rolled LLM judge
python judge_calibration.py --generate  # then hand-grade calibration_set.json yourself
python judge_calibration.py             # check judge agreement with your grading
python ragas_eval.py                    # RAGAS metrics (faithfulness, relevancy, etc.)
```

To prove a change helped, before/after:
```bash
python eval.py --save before.json
# ... make your one change ...
python eval.py --save after.json
python compare_eval_runs.py before.json after.json
```

### Common issues

| Problem | Fix |
|---|---|
| `qdrant: false` in `/health` | Run `docker start qdrant-hr` |
| `OPENROUTER_API_KEY is not set` | Check `.env` exists in the repo root and has the key |
| `ModuleNotFoundError: No module named 'ragas'` (or `sentence_transformers`, `fastapi`, etc.) | Run `pip install -r requirements.txt` inside the activated venv — nothing runs on the system Python |
| Upload/ingest fails with "No text extracted" | File may be a scanned/image-only PDF — OCR isn't supported |
| React page blank / 404 on `/api` | Start FastAPI (port 8000) before the React dev server |
| Confidence always 0% / everything refuses | Lower `CONFIDENCE_THRESHOLD` in `config.py` or the Query page's threshold slider |
| PowerShell `&&` syntax error | Run the two commands separately — PowerShell 5.1 doesn't support `&&` (PowerShell 7+ does) |
| `ragas_eval.py` hangs / times out | Free-tier OpenRouter models can be slow under load — re-run, or check `OPENROUTER_MODEL` in `.env` is still a valid free model |
