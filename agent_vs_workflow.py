"""
Week 7 deliverable: race the hand-built agent (agent.py) against the existing
fixed workflow (structured_answer.ask_structured() -- retrieve once, generate
once) on speed, cost, and reliability.

Deliberately races on the two hardest, most multi-step questions already
proven to be genuinely multi-hop -- reusing eval.py's own regression
QUESTIONS instead of inventing a new test set:
  - "regression: multi-hop reasoning (Week 5 fix)"
  - "regression: known gap - date disambiguation (documented, unresolved)"
These are exactly the "path changes depending on the input" cases the brief
says agents are FOR. Racing on a single-fact lookup question would be an
unfair/uninteresting race -- the fixed workflow should win those trivially.

Cost proxy: number of LLM calls made (OpenRouter free-tier doesn't return
per-call token counts uniformly across fallback models, so call COUNT is used
as the honest, simple proxy -- documented here rather than silently assumed).

Usage:
  python agent_vs_workflow.py                # race both regression questions
  python agent_vs_workflow.py --save race.json
"""
import argparse
import json
import time

import structured_answer
import agent
from eval import QUESTIONS, score
from query import EMBED_MODEL_NAME, QUERY_PREFIX
from vectorstore import get_store
from sentence_transformers import SentenceTransformer
from hybrid import HybridRetriever

COLLECTION = "hr_policy_parent_child"


def _count_calls(module, fn_name):
    """Wrap module.fn_name to count calls, returning (counter_dict, restore_fn)."""
    original = getattr(module, fn_name)
    counter = {"n": 0}

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    setattr(module, fn_name, wrapper)
    return counter, lambda: setattr(module, fn_name, original)


RACE_TAGS = [
    "regression: multi-hop reasoning (Week 5 fix)",
    "regression: known gap - date disambiguation (documented, unresolved)",
]


def race_one(q: dict) -> dict:
    question = q["question"]
    expected_keywords = q["expected_keywords"]

    print(f"\n{'='*80}\nQUESTION: {question}\n{'='*80}")

    # ── Fixed workflow (baseline, already built in earlier weeks) ──────────────
    print("\n--- Fixed workflow (structured_answer.ask_structured) ---")
    counter, restore = _count_calls(structured_answer, "call_openrouter")
    t0 = time.time()
    try:
        store = get_store()
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        hybrid_retriever = HybridRetriever(store, COLLECTION, embed_model, QUERY_PREFIX)
        fixed_result = structured_answer.ask_structured(store, COLLECTION, embed_model, question, hybrid_retriever=hybrid_retriever)
    finally:
        restore()  # always unwrap call_openrouter, even if ask_structured raised
    fixed_elapsed = round(time.time() - t0, 2)
    fixed_calls = counter["n"]
    fixed_answer = fixed_result.answer
    fixed_correct = score(fixed_answer, expected_keywords)
    print(f"Answer: {fixed_answer}")
    print(f"Time: {fixed_elapsed}s | LLM calls: {fixed_calls} | Keyword match: {'PASS' if fixed_correct else 'FAIL'}")

    # ── Agent (this week's new loop) ────────────────────────────────────────────
    print("\n--- Agent (agent.run_agent, ReAct loop) ---")
    agent_result = agent.run_agent(question)
    agent_answer = agent_result["answer"]
    agent_correct = score(agent_answer, expected_keywords)
    print(f"\nAnswer: {agent_answer}")
    print(f"Time: {agent_result['elapsed_seconds']}s | LLM calls: {agent_result['num_llm_calls']} | "
          f"Tool calls: {agent_result['num_tool_calls']} | Steps: {agent_result['num_steps']} | "
          f"Stopped: {agent_result['stopped_reason']} | Keyword match: {'PASS' if agent_correct else 'FAIL'}")

    return {
        "question": question,
        "tag": q["tag"],
        "fixed": {"answer": fixed_answer, "elapsed_seconds": fixed_elapsed, "llm_calls": fixed_calls, "correct": fixed_correct},
        "agent": {"answer": agent_answer, "elapsed_seconds": agent_result["elapsed_seconds"],
                  "llm_calls": agent_result["num_llm_calls"], "tool_calls": agent_result["num_tool_calls"],
                  "steps": agent_result["num_steps"], "stopped_reason": agent_result["stopped_reason"],
                  "correct": agent_correct},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    race_questions = [q for q in QUESTIONS if q["tag"] in RACE_TAGS]
    if not race_questions:
        raise RuntimeError(f"No questions found with tags {RACE_TAGS} -- check eval.py's QUESTIONS list.")

    results = [race_one(q) for q in race_questions]

    print(f"\n\n{'='*100}\nSUMMARY — fixed workflow vs. agent\n{'='*100}")
    print(f"{'Question':<45} {'Fixed (s/calls/ok)':<20} {'Agent (s/calls/ok)':<20} {'Winner':<10}")
    print("-" * 100)
    for r in results:
        f, a = r["fixed"], r["agent"]
        f_str = f"{f['elapsed_seconds']}s/{f['llm_calls']}/{'Y' if f['correct'] else 'N'}"
        a_str = f"{a['elapsed_seconds']}s/{a['llm_calls']}/{'Y' if a['correct'] else 'N'}"
        if f["correct"] and not a["correct"]:
            winner = "fixed"
        elif a["correct"] and not f["correct"]:
            winner = "agent"
        elif f["elapsed_seconds"] <= a["elapsed_seconds"]:
            winner = "fixed (tie/faster)"
        else:
            winner = "agent (tie/faster)"
        print(f"{r['question'][:43]:<45} {f_str:<20} {a_str:<20} {winner:<10}")

    print("-" * 100)
    print("Cost proxy = LLM call count (not token count -- see module docstring for why).")
    print("Read this table, then write down in your own words which one you'd actually ship, and why.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
