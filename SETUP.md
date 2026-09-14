# Setup Guide — RAG HR App (Week 5)

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.13 |
| Node.js | 18+ | For React UI |
| Docker Desktop | any | Runs Qdrant vector store |
| OpenRouter account | — | Free LLM API — https://openrouter.ai/keys |
| Langfuse account | — | Free tracing — https://cloud.langfuse.com |

---

## 1. Clone the repo

```bash
git clone https://github.com/kannanravisoftsuave-svg/rag-hr-app.git
cd rag-hr-app
git checkout rag_Weak5
```

---

## 2. Python environment

```bash
python -m venv venv
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-your-key-here        # from https://openrouter.ai/keys
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free

QDRANT_URL=http://localhost:6333

LANGFUSE_SECRET_KEY=sk-lf-your-secret-key     # from https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com

API_URL=http://localhost:8000
```

---

## 4. Start Qdrant (Docker)

**First time only — creates the container:**
```bash
docker run -d --name qdrant-hr -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Every subsequent session:**
```bash
docker start qdrant-hr
```

Verify it's running: open http://localhost:6333/dashboard

---

## 5. Start the FastAPI backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300
```

Verify: open http://localhost:8000/health — should return `{"status":"ok","qdrant":true}`.

---

## 6. Start the React UI

In a second terminal (venv not required here):

```bash
cd ui-react
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

The Vite dev server proxies `/api/*` → `http://localhost:8000` automatically.

---

## 7. Upload documents and query

1. Open http://localhost:3000 → **Upload** page
2. Drag and drop a PDF, DOCX, MD, or TXT file
3. Select collection (default: `documents`) and chunking strategy (default: `auto`)
4. Click **Upload** — you'll see chunk count and a Langfuse trace link
5. Go to **Query** page, type a question, press Enter
6. Go to **Traces** page to see span breakdown and open Langfuse dashboard

---

## 8. (Optional) Week 6 — evals

For the eval/regression-testing scripts (`eval.py`, `judge.py`, `ragas_eval.py`,
etc.), see **[EVALS_SETUP.md](EVALS_SETUP.md)** — it needs one extra
`pip install` for the RAGAS-based script, explained there.

---

## 9. (Optional) Streamlit UI

If you prefer the Streamlit interface:

```bash
streamlit run ui/app.py
```

Opens at http://localhost:8501.

---

## Architecture overview

```
Browser (React :3000)
    │  /api/* proxy
    ▼
FastAPI (:8000)
    ├─ POST /upload   → ingestion/pipeline.py → Qdrant
    ├─ POST /query    → vectorstore → rerank → OpenRouter LLM
    ├─ GET  /documents / /chunks / /collections
    └─ GET  /health
         │
         ├─ Qdrant (:6333)  — vector store (BGE-small-en-v1.5, 384-dim, cosine)
         ├─ OpenRouter       — hosted LLM (nemotron-3.5-lightning:free + fallbacks)
         └─ Langfuse Cloud   — trace/span observability
```

---

## Common issues

| Problem | Fix |
|---|---|
| `qdrant: false` in health | Run `docker start qdrant-hr` |
| `OPENROUTER_API_KEY is not set` | Check `.env` file is in the repo root |
| Upload fails with "No text extracted" | File may be a scanned/image-only PDF; OCR not supported |
| React page blank / 404 on `/api` | Make sure FastAPI is running on port 8000 first |
| Confidence always 0% | Lower the threshold slider in the Query settings sidebar |
| PowerShell `&&` syntax error | Run commands separately — PowerShell 5.1 doesn't support `&&` |
