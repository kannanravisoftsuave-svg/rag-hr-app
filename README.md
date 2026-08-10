# Ask My HR Documents — Mini RAG App

A local, free "ask my documents" app for HR policy: ingests a base employee
handbook plus a policy addendum, answers questions using only those
documents with a citation, and says "I don't know" if the answer isn't
in the documents.

**Stack:** Python + `sentence-transformers` (BGE-small embeddings, local) +
Chroma (local vector store) + Ollama (local LLM, `llama3.2:3b`). No API
keys, no cost.

## Project layout

```
data/
  employee_handbook.md     base HR policy (v3.0)
  policy_addendum.md       new policy addendum being ingested
chroma_db/                 persisted vector store (created by ingest.py)
ingest.py                  builds two chunking-strategy collections
query.py                   interactive CLI to ask questions
eval.py                    runs a fixed question set against both collections
FINDINGS.md                write-up: chunk-size comparison + failure modes
```

## One-time setup

Already done in this environment:
- Python 3.13 venv at `venv/`
- Packages installed: `sentence-transformers`, `chromadb`, `pypdf`
- Ollama installed with model `llama3.2:3b` pulled

To set up from scratch elsewhere (run in `cmd.exe`, from inside `rag-hr-app\`):
```bat
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install sentence-transformers chromadb pypdf
ollama pull llama3.2:3b
```

## Running it

All commands below assume you're in `cmd.exe` with the current directory set to
`D:\AI\rag-hr-app`, e.g.:
```bat
cd D:\AI\rag-hr-app
```

### 1. Make sure Ollama is running
```bat
curl -s http://localhost:11434/api/version
```
If that fails, start the Ollama app, or run `ollama serve` in a terminal.

### 2. Ingest the documents
Run this whenever files in `data\` change:
```bat
venv\Scripts\python.exe ingest.py
```
Builds two Chroma collections at different chunk granularities:
- `hr_policy_subsection` — one chunk per `###` sub-heading (small, precise)
- `hr_policy_section` — one chunk per `##` heading, merging its sub-headings (larger, more context)

Check the printed chunk counts per source file — a `0 chunks` line for any
file means the chunker silently dropped that document (this happened once
during development; see `FINDINGS.md` §5).

### 3. Ask questions interactively
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

REM use a different Ollama model (pull it first: ollama pull <name>)
venv\Scripts\python.exe query.py --model llama3.1:8b
```

### 4. Run the chunk-size comparison
```bat
venv\Scripts\python.exe eval.py
```
Runs the same question set against both collections back-to-back so you
can compare retrieval/answer quality side by side. Edit the `QUESTIONS`
list in `eval.py` to add your own test cases.

## Known limitations (see FINDINGS.md for full detail)

- The small local model (`llama3.2:3b`) can misread or fail to extract a
  fact from a large/dense chunk even when retrieval finds the right one —
  smaller chunks reduce this.
- Multi-hop questions requiring two facts from different sections can
  fail at the reasoning/synthesis step, not just retrieval.
- On questions the documents are genuinely silent about (not simply
  "missing," but never addressing the specific scenario), the model
  sometimes manufactures a confident yes/no answer instead of saying so —
  a bias toward giving *a* definitive answer over admitting silence.
