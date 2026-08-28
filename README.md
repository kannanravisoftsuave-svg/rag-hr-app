# RAG HR App — Week 5: Dynamic RAG with Observability

An end-to-end Retrieval-Augmented Generation (RAG) app for HR policy documents. Upload PDFs, DOCX, Markdown, or plain-text files, ask natural-language questions, and get cited answers — with full LLM observability via Langfuse.

**Stack:** FastAPI · Qdrant · BGE-small embeddings · Cross-encoder reranking · OpenRouter LLM · Langfuse tracing · React + Vite + Tailwind CSS

---

## What was built

### Week 1–3 (foundation)
- Python CLI (`query.py`) with BGE-small-en-v1.5 embeddings stored in Qdrant
- Sliding-window and heading-based chunking strategies
- Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Confidence gate — blocks LLM call when best similarity is below threshold
- Hybrid BM25 + dense vector retrieval
- OpenRouter free LLM with fallback model chain

### Week 4 (API + Tracing)
- **FastAPI backend** (`api/`) replacing the CLI — endpoints for upload, query, documents, chunks, collections, health
- **Langfuse v4 tracing** (`tracing.py`) — every query and upload creates a root span with nested child spans:
  - Query: `retrieval` → `reranking` → `confidence_gate` → `llm_generation`
  - Ingestion: `text_extraction` → `chunking` → `embedding` → `vector_store`
- Pydantic v2 request/response models (`api/models.py`)
- Streamlit UI (`ui/app.py`) with file upload, query, document browser

### Week 5 (React UI + Pipeline hardening)
- **React UI** (`ui-react/`) — replaced Streamlit with Vite + React + Tailwind CSS:
  - **Upload page** — drag-and-drop, collection selector, chunking strategy, metrics, Langfuse link
  - **Query page** — settings sidebar (collection, filter by doc, top-k, threshold, hybrid toggle), answer panel with confidence badge, expandable chunk list, trace link
  - **Documents page** — collection picker, expandable document list with chunk explorer
  - **Traces page** — span reference table, Langfuse dashboard link, last-query trace card
- **Pipeline hardening:**
  - UUID-based Qdrant point IDs (fixed integer ID collision bug when re-uploading files)
  - Batch embedding (32 chunks at a time — prevents OOM on large documents)
  - Sigmoid(CE score) as confidence signal — cross-encoder score is a better relevance indicator than raw dense similarity
  - `_delete_by_source()` deduplication — clean re-upload without stale chunks
  - `.txt` file support added
- **Confidence threshold** lowered to `0.45` (re-calibrated for short/unpunctuated text like org charts)
- **Chunker improvements** — hard-splits at 400 chars even without punctuation; splits on `\n` for org chart / table text

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
```

---

## Quick start

See **[SETUP.md](SETUP.md)** for the full setup guide.

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
