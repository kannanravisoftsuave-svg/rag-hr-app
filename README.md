# RAG HR App — Week 5: Dynamic RAG with Observability

An end-to-end Retrieval-Augmented Generation (RAG) app for HR policy documents. Upload PDFs, DOCX, Markdown, or plain-text files, ask natural-language questions, and get cited answers — with full LLM observability via Langfuse.

**Stack:** FastAPI · Qdrant · BGE-small embeddings · Cross-encoder reranking · OpenRouter LLM · Langfuse tracing · React + Vite + Tailwind CSS

---

## What was built — Week 1 to Week 6

This project follows a 6-week AI Engineering curriculum (Module 1–3: Foundations →
Prompting/Tool Calling → Retrieval & RAG → Debugging Retrieval → Error Analysis → Evals).
Below is what was actually delivered each week, against what each week asked for.

### Week 1 — Foundations (theory only, no code deliverable)
Covered how language models, tokens, temperature, and embeddings actually work — no app code
this week by design. Concepts from this week show up in later code as: `temperature=0.1` for
consistent factual answers and a multi-model `FALLBACK_MODELS` chain (`query.py`).
**Known gap:** token usage / cost per query is not yet logged to Langfuse, even though the
tracing infrastructure (`tracing.py`) already exists to carry it.

### Week 2 — Prompting, Structured Output & Tool Calling
- Original prompt (`query.py` `SYSTEM_PROMPT`) asks the model for free-text `Quotes:` /
  `Reasoning:` / `Answer:` sections — functional, but nothing validated the model actually
  complied.
- **Retrofitted this session:** `structured_answer.py` — a Pydantic `HRAnswer` schema
  (`quotes`, `reasoning`, `answer`, `sources`, `is_refusal`) + OpenRouter JSON mode, with
  validation-error-fed-back retry on malformed output. Additive, not a replacement — opt in via
  `python query.py --structured` or `use_structured_output: true` on `POST /query`.
- Tool calling was not implemented — this app is pure retrieve-then-generate, so there was no
  real action for the model to trigger. Noted as a deliberate scope choice, not an oversight.

### Week 3 — Retrieval & RAG (build week)
- Document ingestion (`ingestion/pipeline.py`, `extractor.py`) — PDF / DOCX / MD / TXT
- Structure-aware Markdown-heading chunking (`ingestion/chunker.py`), compared at two
  granularities (`hr_policy_subsection` vs `hr_policy_section`) — see `FINDINGS.md` §2
- Parent-child chunking added later (`hr_policy_parent_child`) — embeds small precise chunks,
  returns the larger parent section to the LLM for context, resolving the precision-vs-context
  tradeoff between the two granularities above
- Qdrant vector store (`vectorstore.py`), grounded generation with citations, and a
  confidence-threshold gate that refuses instead of guessing when nothing relevant is found

### Week 4 — Debugging Retrieval (hybrid, reranking, failure separation)
- `failure_label()` in `eval.py` — classifies every result as `RETRIEVAL FAILURE` (wrong
  document fetched) vs `GENERATION FAILURE` (right document, bad answer) vs `OK`
- `hybrid.py` — BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) in `query.py`
- `compare_hitrate()` in `eval.py` — before/after hit-rate@3 table proving the hybrid change
  helped, with no LLM calls needed
- `debug.py` — inspection view showing retrieved chunks + answer + diagnosis side by side

### Week 5 — Error Analysis (reading traces like a professional)
- `FINDINGS.md` — ~20 real traces read by hand, honest per-failure notes, grouped into a named,
  ranked failure taxonomy
- A single multi-hop test question exposed **five separable failure modes** at different fix
  points (retrieval → extraction → reasoning → model-capability ceiling → prompt design) — each
  isolated and fixed independently, with one residual gap (date disambiguation) documented as
  unresolved rather than hidden
- `questions.md` — categorized sample question bank (basic facts, addendum-supersedes,
  refusals, multi-hop, known-shaky edge cases)

### Week 6 — Evals (measuring whether a change actually helped)
- `eval.py` — one-command test set: keyword-match scoring, retrieval hit-rate@3, failure
  classification, results grouped by problem-type tag
- **Regression tests** — the real Week 5 failures turned into permanent test cases, so a fix
  can't silently regress unnoticed
- `judge.py` + `judge_calibration.py` — hand-rolled LLM-as-judge for questions a keyword match
  can't score well, validated against human grading before being trusted (never assumed correct)
- `ragas_eval.py` — the same question set scored with RAGAS's pre-built metrics (faithfulness,
  answer relevancy, context precision/recall) as an alternative to the hand-rolled judge
- `compare_eval_runs.py` — before/after score comparison per problem type

### Also built (infrastructure spanning multiple weeks)
- **FastAPI backend** (`api/`) — endpoints for upload, query, documents, chunks, collections, health
- **React UI** (`ui-react/`) — upload, query, document browser, and trace-viewer pages;
  Streamlit UI (`ui/app.py`) kept as a legacy fallback
- **Langfuse v4 tracing** (`tracing.py`) — every query/upload creates a root span with nested
  child spans (`retrieval` → `reranking` → `confidence_gate` → `llm_generation`, and
  `text_extraction` → `chunking` → `embedding` → `vector_store` for ingestion)
- Pipeline hardening — UUID-based Qdrant point IDs, batched embedding, dedup on re-upload,
  sigmoid(cross-encoder score) as a better confidence signal than raw dense similarity

---

## Project layout

```
api/
  main.py          FastAPI app — all endpoints
  models.py        Pydantic request/response models
ingestion/
  pipeline.py      File → Qdrant ingestion (extract → chunk → embed → store)
  extractor.py     PDF / DOCX / MD / TXT text extraction
  chunker.py       Sliding-window, heading, page chunking strategies
ui-react/
  src/
    App.jsx        Sidebar nav + health badge
    api.js         axios client (proxied to :8000)
    pages/
      Upload.jsx
      Query.jsx
      Documents.jsx
      Traces.jsx
ui/
  app.py           Streamlit UI (legacy, still works)
config.py          All settings (model names, threshold, collection names)
tracing.py         Langfuse v4 span/generation helpers
vectorstore.py     Qdrant wrapper (query, upsert, list, count)
hybrid.py          BM25 + dense hybrid retrieval
query.py           CLI entrypoint + build_prompt / SYSTEM_PROMPT
.env.example       Template — copy to .env and fill in your keys
SETUP.md           Full setup instructions
eval.py            Week 6: rule-based eval + regression tests + retrieval before/after
judge.py           Week 6: hand-rolled LLM-as-judge
judge_calibration.py  Week 6: validates judge.py against your own grading
ragas_eval.py      Week 6: same eval set, scored with RAGAS instead
compare_eval_runs.py  Week 6: diff two saved eval.py runs, per problem type
EVALS_SETUP.md     Week 6: setup for the eval scripts above
```

---

## Quick start

See **[SETUP.md](SETUP.md)** for the full setup guide, then **[EVALS_SETUP.md](EVALS_SETUP.md)** for the Week 6 eval scripts.

```bash
# 1. Start Qdrant
docker start qdrant-hr

# 2. Start FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

# 3. Start React UI (separate terminal)
cd ui-react && npm run dev

# 4. Open http://localhost:3000
```

---

## Configuration

All tunable settings are in `config.py`. Key values:

| Setting | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.45` | Dense similarity below this blocks the LLM call |
| `DEFAULT_TOP_K` | `8` | Chunks retrieved per query |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Local embedding model (384-dim) |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3.5-lightning:free` | Primary LLM (5 fallbacks configured) |

---

## Observability

Every query and upload is traced in Langfuse. Each trace has:

| Span | Records |
|---|---|
| `retrieval` | Hit count, top similarity scores |
| `reranking` | CE score, sigmoid confidence |
| `confidence_gate` | Dense sim vs threshold, pass/fail |
| `llm_generation` | Full prompt, model response |
| `text_extraction` | Format, char count, table detection |
| `chunking` | Strategy, chunk count, avg size |
| `embedding` | Model, batch count |
| `vector_store` | Points stored, old points deleted |

View traces at https://cloud.langfuse.com after configuring `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` in `.env`.
