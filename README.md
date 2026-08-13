# Ask My HR Documents — Mini RAG App

An "ask my documents" app for HR policy: ingests a base employee handbook
plus a policy addendum, answers questions using only those documents with
a citation, and says "I don't know" if the answer isn't in the documents.

**Stack:** Python + `sentence-transformers` (BGE-small embeddings, local,
free) + **Qdrant running in Docker** (vector store, server mode) +
**OpenRouter** (hosted LLM, `google/gemma-4-26b-a4b-it:free` by default —
free tier, needs an API key).

See `VERSIONS.md` for how this evolved (v1: fully local/Ollama/Chroma →
v2: OpenRouter + Qdrant added, bug fixes, auto-scoring → v3: Chroma
removed, Qdrant moved into Docker).

## Project layout

```
data/
  employee_handbook.md     base HR policy (v3.0)
  policy_addendum.md       new policy addendum being ingested
vectorstore.py             wrapper over Qdrant (build/query/count collections)
ingest.py                  builds two chunking-strategy collections
query.py                   interactive CLI to ask questions
eval.py                    runs a fixed question set against both collections, auto-scored
inspect_db.py              peek at stored chunks/metadata/embeddings
FINDINGS.md                write-up: chunk-size comparison + failure modes
VERSIONS.md                version history: what changed and why, v1 -> v3
questions.md               categorized sample questions to try
```

## One-time setup

Already done in this environment:
- Python 3.13 venv at `venv/`
- Packages installed: `sentence-transformers`, `qdrant-client`, `pypdf`, `requests`
- Docker Desktop running, Qdrant server container `qdrant-hr` created

To set up from scratch elsewhere (run in `cmd.exe`, from inside `rag-hr-app\`):
```bat
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install sentence-transformers qdrant-client pypdf requests
docker run -d --name qdrant-hr -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Start Qdrant before every session (if not already running)
```bat
docker start qdrant-hr
```
Check it's up: open http://localhost:6333/dashboard in a browser, or:
```bat
curl http://localhost:6333
```
(Only use `docker run` the very first time — it creates the container.
Every time after that, use `docker start qdrant-hr` to reuse the same
container and keep its data.)

### Get an OpenRouter API key (required for generation)
1. Sign up free at https://openrouter.ai
2. Go to https://openrouter.ai/keys and create a key
3. Set it as an environment variable (don't hardcode it anywhere):
```bat
set OPENROUTER_API_KEY=sk-or-your-key-here
```
This only lasts for the current `cmd.exe` session. To persist it across
sessions, use `setx OPENROUTER_API_KEY "sk-or-your-key-here"` and open a new
terminal afterward.

## Running it

All commands below assume you're in `cmd.exe` with the current directory set to
`D:\AI\rag-hr-app`, e.g.:
```bat
cd D:\AI\rag-hr-app
```

### 1. Ingest the documents
Run this whenever files in `data\` change:
```bat
venv\Scripts\python.exe ingest.py
```
Builds two collections at different chunk granularities:
- `hr_policy_subsection` — one chunk per `###` sub-heading (small, precise)
- `hr_policy_section` — one chunk per `##` heading, merging its sub-headings (larger, more context)

Check the printed chunk counts per source file — a `0 chunks` line for any
file means the chunker silently dropped that document (this happened once
during development; see `FINDINGS.md` §5).

### 2. Ask questions interactively
```bat
venv\Scripts\python.exe query.py
```
Type a question at the `Q>` prompt, `exit` to quit.

Useful flags:
```bat
REM use the larger-chunk collection instead of the default (hr_policy_section)
venv\Scripts\python.exe query.py --collection hr_policy_subsection

REM retrieve more candidate chunks before generation (default is 4)
venv\Scripts\python.exe query.py --top_k 8

REM use a different OpenRouter model
venv\Scripts\python.exe query.py --model meta-llama/llama-3.1-8b-instruct:free

REM change or disable the confidence-threshold auto-refuse (default 0.70, see below)
venv\Scripts\python.exe query.py --threshold 0.6
venv\Scripts\python.exe query.py --threshold 0
```
Browse free/paid model names at https://openrouter.ai/models (filter by price).

### 3. Run the chunk-size comparison (now auto-scored)
```bat
venv\Scripts\python.exe eval.py
```
Runs the fixed question set (in `eval.py`'s `QUESTIONS` list) against both
collections, automatically scoring each answer PASS/FAIL against expected
keywords, and prints a final score summary per collection — no manual
reading required. Edit `QUESTIONS` to add your own test cases.

### Peek at what's actually stored
```bat
venv\Scripts\python.exe inspect_db.py hr_policy_subsection 5
```
Shows raw chunk text, metadata, and a peek at the embedding vector for
the first N chunks in a collection.

## How retrieval confidence works

`query.py` checks the best-match similarity score before calling the LLM
at all. Below `CONFIDENCE_THRESHOLD` (0.70, set in `query.py`), it refuses
immediately with no network call — deterministic, not dependent on the
LLM's own judgment. This threshold was calibrated empirically: on a
10-question sample, correct answers' top-hit similarity clustered
0.78-0.90, genuinely unanswerable questions clustered 0.55-0.65. Re-tune
via `--threshold` if it misfires on questions outside that sample.

## Known limitations (see FINDINGS.md for full detail)

- A small/weak generation model can misread or fail to extract a fact from
  a large/dense chunk even when retrieval finds the right one — smaller
  chunks reduce this.
- Multi-hop questions requiring two facts from different sections can fail
  at the reasoning/synthesis step, not just retrieval.
- On questions the documents are genuinely silent about (not simply
  "missing," but never addressing the specific scenario), a model can
  manufacture a confident yes/no answer instead of saying so — a bias
  toward giving *a* definitive answer over admitting silence.
- Generation is hosted, so `query.py`/`eval.py` need network access and a
  valid `OPENROUTER_API_KEY`; retrieval/embeddings remain fully local and
  offline-capable.
- The confidence threshold is calibrated on a small (10-question) sample —
  treat it as a reasonable default, not a proven-optimal cutoff.
