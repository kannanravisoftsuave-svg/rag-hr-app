# Evals Setup Guide — Week 6

This covers the four new scripts added for Week 6 (evals & error analysis).
For the base app (Qdrant, FastAPI, React), see [SETUP.md](SETUP.md) first —
these scripts assume that setup is already done and working.

| Script | What it does | Needs a new install? |
|---|---|---|
| [eval.py](eval.py) | Rule-based scoring (keyword match) + retrieval hit-rate + regression tests | No — already in `requirements.txt` |
| [judge.py](judge.py) | Hand-rolled LLM-as-judge, called by `eval.py --judge` | No |
| [judge_calibration.py](judge_calibration.py) | Checks the hand-rolled judge agrees with your own grading | No |
| [ragas_eval.py](ragas_eval.py) | Same questions, scored with RAGAS's pre-built metrics instead | **Yes — see below** |

---

## 1. Install the new dependencies (only needed for `ragas_eval.py`)

```bash
pip install -r requirements.txt
```

This pulls in three new packages added to `requirements.txt` for Week 6:

| Package | Why it's needed |
|---|---|
| `ragas` | The eval library itself — ships the faithfulness / answer_relevancy / context_precision / context_recall metrics |
| `langchain-openai` | RAGAS expects a LangChain-style chat model, not a raw `requests.post()` call. OpenRouter speaks the same API format as OpenAI, so this wraps it by pointing `base_url` at OpenRouter instead of OpenAI — no separate OpenAI account needed |
| `langchain-huggingface` | Lets RAGAS use the SAME local BGE embedding model (`BAAI/bge-small-en-v1.5`) already used for retrieval, instead of requiring a second embeddings API |

This install can take a few minutes and pulls in `torch`/`transformers`-adjacent
packages if they aren't already present (they should be, since
`sentence-transformers` already depends on them).

---

## 2. No new environment variables

`ragas_eval.py` reuses the same `OPENROUTER_API_KEY` already in your `.env`
(see [SETUP.md §3](SETUP.md)) — nothing new to add there.

---

## 3. Running the evals

Make sure Qdrant is running and documents are already ingested (same
prerequisite as `query.py` — see [SETUP.md §4](SETUP.md)).

**Rule-based + retrieval eval (fast, free, no LLM judge):**
```bash
python eval.py
```

**Same, plus the hand-rolled LLM judge on the two rubric-graded questions:**
```bash
python eval.py --judge
```
> Only trust this after running the calibration check below at least once.

**Validate the hand-rolled judge against your own grading:**
```bash
python judge_calibration.py --generate
# open calibration_set.json, read each answer, fill in "human_score": 1-5 yourself
python judge_calibration.py
```

**RAGAS eval (all questions, 4 pre-built metrics):**
```bash
python ragas_eval.py
```
This makes real LLM calls — 4 metrics × ~10 questions ≈ 40 OpenRouter calls per
run. Expect it to take a few minutes on the free-tier models.

**Save a run and compare before/after a change:**
```bash
python eval.py --save before.json
# ... make your one improvement ...
python eval.py --save after.json
python compare_eval_runs.py before.json after.json
```

---

## 4. What "good" looks like

- `eval.py` — pass rate per tag close to 100%, `hit-rate@3` close to 100%. The
  `known gap` regression row is EXPECTED to still fail (it's a documented,
  unresolved issue from `FINDINGS.md`) — that's normal, not a bug in the eval.
- `judge_calibration.py` — "within 1 point" agreement ≥ 80% before you trust
  `eval.py --judge`'s numbers for anything.
- `ragas_eval.py` — `faithfulness` and `answer_relevancy` close to 1.0 for
  in-scope questions (the out-of-scope refusal question will score low on
  these since there's no real answer to be "faithful" to — that's expected).
  `context_precision`/`context_recall` close to 1.0 means retrieval is finding
  the right, and only the right, material.

---

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ragas'` | Run `pip install -r requirements.txt` — the Week 6 deps are new |
| `OPENROUTER_API_KEY is not set` | Check `.env` exists in repo root and has the key (see SETUP.md §3) |
| `ragas_eval.py` hangs or times out | OpenRouter free-tier models can be slow/rate-limited under load; re-run, or check `query.py`'s `FALLBACK_MODELS` list is still valid |
| RAGAS scores all `None`/`NaN` | Usually means the LLM's graded output didn't parse — check the OpenRouter model in `.env` (`OPENROUTER_MODEL`) is still available/free |
| `judge_calibration.py` says "not yet trusted" | Expected the first time — read `judge.py`'s prompt, tighten the rubric wording in `eval.py`'s `QUESTIONS`, and recalibrate |
