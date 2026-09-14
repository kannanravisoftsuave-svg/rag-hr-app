"""
Validate the LLM judge before trusting its scores.

An AI judge you never checked is just a confident number nobody trusts. This
script runs judge.judge() on a handful of real answers you have ALREADY graded
yourself, and reports how often the judge agrees with you.

Workflow:
  1. python judge_calibration.py --generate
       Asks the rubric-graded questions in eval.py for real, and writes the
       question/rubric/answer into calibration_set.json with human_score left
       blank (null).
  2. Open calibration_set.json and fill in "human_score" (1-5) for each entry
     yourself -- read the answer and grade it the way you'd grade a person.
  3. python judge_calibration.py
       Runs judge.judge() on the same answers and compares to your scores.

Rule of thumb: if the judge isn't within 1 point of you on at least ~80% of
the calibration set, don't trust its number yet -- fix the rubric or the
judge prompt and recalibrate before using --judge in eval.py.
"""
import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from vectorstore import get_store
from query import ask, EMBED_MODEL_NAME, QUERY_PREFIX
from eval import QUESTIONS
from judge import judge

CALIBRATION_FILE = Path(__file__).parent / "calibration_set.json"


def generate(collection="hr_policy_parent_child", top_k=8):
    """Ask every rubric-graded question for real and save question/rubric/answer,
    with human_score left blank for you to fill in by hand."""
    rubric_questions = [q for q in QUESTIONS if q.get("rubric")]
    if not rubric_questions:
        print("No rubric-graded questions found in eval.py QUESTIONS. Add a 'rubric' "
              "field to a question before generating a calibration set.")
        return

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    from hybrid import HybridRetriever
    hybrid = HybridRetriever(store, collection, model, QUERY_PREFIX)

    entries = []
    for q in rubric_questions:
        print(f"Asking: {q['question']}")
        answer, _ = ask(store, collection, model, q["question"], top_k=top_k,
                         verbose=False, hybrid_retriever=hybrid)
        entries.append({
            "question": q["question"],
            "rubric": q["rubric"],
            "answer": answer,
            "human_score": None,  # <-- fill this in yourself: 1-5
        })

    CALIBRATION_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nWrote {len(entries)} entries to {CALIBRATION_FILE}.")
    print("Open it, read each answer, and fill in \"human_score\" (1-5) yourself.")
    print("Then re-run: python judge_calibration.py")


def compare():
    if not CALIBRATION_FILE.exists():
        print(f"{CALIBRATION_FILE} not found. Run with --generate first.")
        return

    entries = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    graded = [e for e in entries if e.get("human_score") is not None]
    ungraded = len(entries) - len(graded)
    if ungraded:
        print(f"({ungraded} entries still have human_score=null -- skipping those)")
    if not graded:
        print("No graded entries yet. Fill in human_score in calibration_set.json first.")
        return

    print(f"\n{'Question':<55} {'Human':>6} {'Judge':>6} {'Diff':>5}")
    print("-" * 80)

    exact, within_1, unparseable = 0, 0, 0
    for e in graded:
        result = judge(e["question"], e["rubric"], e["answer"])
        score = result["score"]
        if score is None:
            unparseable += 1
            print(f"{e['question'][:53]:<55} {e['human_score']:>6} {'ERR':>6}   {result['reason'][:40]}")
            continue
        diff = abs(score - e["human_score"])
        exact += diff == 0
        within_1 += diff <= 1
        print(f"{e['question'][:53]:<55} {e['human_score']:>6} {score:>6} {diff:>5}")

    n = len(graded)
    scored = n - unparseable
    print("-" * 80)
    print(f"Graded entries       : {n}")
    if unparseable:
        print(f"Judge parse failures : {unparseable}  (fix judge.py's prompt/parsing before trusting it)")
    if scored:
        print(f"Exact agreement      : {exact}/{scored}  ({exact/scored:.0%})")
        print(f"Within 1 point       : {within_1}/{scored}  ({within_1/scored:.0%})")
        verdict = "TRUSTED" if within_1 / scored >= 0.8 else "NOT YET TRUSTED"
        print(f"\nVerdict: {verdict} "
              f"({'>=80% within 1 point' if within_1/scored >= 0.8 else '<80% within 1 point -- revise the rubric or judge prompt and recalibrate'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="ask real questions and write a fresh calibration_set.json for you to grade")
    args = parser.parse_args()

    if args.generate:
        generate()
    else:
        compare()
