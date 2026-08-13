# Version History

## v1 — Fully local, free
- **Vector store:** Chroma (local/embedded, no server)
- **Generation:** Ollama, local (`llama3.2:3b`, later also tested `llama3.1:8b`)
- **Chunking:** two structure-aware granularities — `hr_policy_subsection` (per `###`) and `hr_policy_section` (per `##`)
- **Prompt:** grounded generation with citations, refined over several rounds (Quotes → Reasoning → Answer structure added to fix real misreading/overreach bugs found during testing)
- **Findings:** chunk-size comparison, and a 6-mode failure taxonomy (retrieval failure, extraction failure, reasoning failure, model-capability ceiling, prompt-design failure, overreach-from-silence) — see `FINDINGS.md`
- Everything ran offline, $0 cost, no API keys

## v2 — Hosted generation, second vector DB, automated scoring
- **Generation switched to OpenRouter** (hosted, free-tier model `google/gemma-4-26b-a4b-it:free`) — removed Ollama entirely, freed local disk
- **Added Qdrant** as a second vector store alongside Chroma, to compare a second backend from the syllabus's named options (Qdrant/Chroma/pgvector)
- **Caught and fixed a real bug:** Chroma was defaulting to squared-L2 distance, not cosine, so its "similarity" numbers were silently wrong this whole time. Fixed by setting `metadata={"hnsw:space": "cosine"}` explicitly.
- **Added a confidence-threshold auto-refuse** in `query.py`: below a similarity of 0.70 (calibrated empirically after the cosine fix), the app skips the LLM call entirely and refuses deterministically, instead of relying on the model's own judgment every time
- **Added automated scoring** in `eval.py`: each question now has expected keywords, and the script prints PASS/FAIL + a score summary automatically instead of requiring a human to read every answer

## v3 — Qdrant only, Qdrant in Docker
- **Removed Chroma entirely** — code, dependency, and data — once Qdrant was confirmed to match it exactly on retrieval quality (same top chunks, same similarity scores after the cosine fix). `vectorstore.py` is now a single `QdrantStore` class, no `--backend` flag anywhere.
- **Switched Qdrant from local/embedded mode to a real server running in Docker** (`docker run ... qdrant/qdrant`, container name `qdrant-hr`), so collections are browsable in Qdrant's web dashboard at `localhost:6333/dashboard` instead of only via `inspect_db.py`
- Re-ingested and re-verified retrieval gives identical results to local mode — confirms the backend swap changed nothing about answer quality, only where/how the data is served

## Current stack (as of v3)
| Component | What |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local, free) |
| Vector store | Qdrant, running in Docker (`localhost:6333`) |
| Generation | OpenRouter, `google/gemma-4-26b-a4b-it:free` (hosted, free tier, needs `OPENROUTER_API_KEY`) |
| Confidence gate | Similarity < 0.70 → auto-refuse, no LLM call |
| Chunking | Structure-aware, two granularities (`hr_policy_subsection`, `hr_policy_section`) |
