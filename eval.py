"""
Evaluation: automated scoring + failure classification + hit-rate@k.

Each question carries expected answer keywords (for answer correctness) and
expected source documents (for retrieval quality). Failures are classified as:
  RETRIEVAL FAILURE  — right document never appeared in top-k
  GENERATION FAILURE — right document was retrieved, but answer was wrong
  OK                 — retrieved correctly and answered correctly

Also measures hit-rate@3 (did the right document appear in the top 3 results?)
before (dense only) and after (hybrid BM25+dense) to prove the hybrid change helped.

Results are grouped by "tag" (problem type), so you can see which category of
question broke or improved -- not just one overall number.

Usage:
  python eval.py                          # full run: both collections + before/after compare
  python eval.py --no-compare             # skip the retrieval before/after comparison (faster)
  python eval.py --judge                  # also run the LLM judge on rubric-graded questions
                                           #   (only trust this after judge_calibration.py passes)
  python eval.py --save results.json      # save per-tag scores for later comparison
                                           #   (see compare_eval_runs.py)
"""
import argparse
import json
from sentence_transformers import SentenceTransformer
from vectorstore import get_store
from query import ask, retrieve, EMBED_MODEL_NAME, QUERY_PREFIX

# Each question is a dict:
#   question           the question text
#   tag                problem type / category — used to group before/after scores
#   expected_keywords  PASS if ANY keyword appears in the answer (case-insensitive)
#   expected_sources   source filenames that MUST appear in top-k for retrieval to
#                       count as a hit. Empty list = out-of-scope question (no retrieval check)
#   rubric             optional. If set, this question is ALSO graded by the LLM judge
#                       (judge.py) with --judge, for cases a keyword match can't score well
#                       (multi-part synthesis, "did it reach the right conclusion").
#   reference           a short ground-truth answer, in plain sentences. Only used by
#                       ragas_eval.py (RAGAS's context_precision/context_recall metrics need
#                       something to compare retrieval against) — eval.py itself ignores it.
QUESTIONS = [
    dict(
        question="How many days of paid sick leave do employees get per year?",
        tag="basic fact",
        expected_keywords=["10 days", "10 paid days"],
        expected_sources=["employee_handbook.md"],
        reference="Employees get 10 days of paid sick leave per year.",
    ),
    dict(
        question="How many weeks of paid parental leave does a primary caregiver get?",
        tag="addendum supersedes handbook",
        expected_keywords=["16 weeks"],
        expected_sources=["policy_addendum.md"],
        reference=(
            "Primary caregivers get 16 weeks of paid parental leave, per the addendum, "
            "which supersedes the handbook's earlier figure."
        ),
    ),
    dict(
        question="What is the home-office equipment allowance?",
        tag="addendum supersedes handbook",
        expected_keywords=["$500", "500"],
        expected_sources=["policy_addendum.md"],
        reference="The home-office equipment allowance is $500, per the addendum.",
    ),
    dict(
        question="How many days per week can eligible employees work remotely?",
        tag="addendum supersedes handbook",
        expected_keywords=["3 days", "three days"],
        expected_sources=["policy_addendum.md"],
        reference="Eligible employees can work remotely 3 days per week, per the addendum.",
    ),
    dict(
        question="What is the probation period for new hires?",
        tag="basic fact, handbook only",
        expected_keywords=["90-day", "90 day", "90 days"],
        expected_sources=["employee_handbook.md"],
        reference="The probation period for new hires is 90 days.",
    ),
    dict(
        question="Can an employee paste customer data into an unapproved AI tool?",
        tag="regression: chunk dilution (Week 5 fix)",
        expected_keywords=["no", "not", "must never", "confidential"],
        expected_sources=["policy_addendum.md"],
        rubric=(
            "Must say employees may NOT paste customer data into an unapproved AI tool, "
            "and must cite the addendum's data-confidentiality rule as the basis."
        ),
        reference=(
            "No, employees must never paste customer data into an unapproved AI tool -- "
            "this is a confidentiality violation under the addendum."
        ),
    ),
    dict(
        question="What is the company's policy on stock options for employees?",
        tag="NOT in documents -> should refuse",
        expected_keywords=["don't know", "not mentioned", "no information", "does not"],
        expected_sources=[],  # out-of-scope: no expected source
        reference=(
            "The documents do not mention a stock options policy, so the assistant should "
            "say it has no information on this rather than guessing."
        ),
    ),
    dict(
        question="What is the annual performance bonus cap?",
        tag="basic fact, handbook only",
        expected_keywords=["10%"],
        expected_sources=["employee_handbook.md"],
        reference="The annual performance bonus cap is 10% of salary.",
    ),
    # --- Regression tests from Week 5's manual error analysis (see FINDINGS.md) ---
    # Each of these is a REAL failure that was found by hand and then fixed. Keeping the
    # question here permanently means the fix (top_k, chunking, or prompt change) can never
    # silently regress without an eval run catching it.
    dict(
        question=(
            "A new employee is 45 days into their job and wants to start working remotely "
            "2 days a week. Is that allowed right now, and if not, when would it become allowed?"
        ),
        tag="regression: multi-hop reasoning (Week 5 fix)",
        # Keyword check alone is a weak signal here (the failure mode was that the model
        # *contradicted itself* between a correct Reasoning section and a wrong Answer
        # section -- a keyword match on the final Answer text can miss that). The rubric
        # lets --judge catch a right-conclusion-wrong-answer contradiction that keywords can't.
        expected_keywords=["90", "not eligible", "not allowed", "once", "after"],
        expected_sources=["employee_handbook.md", "policy_addendum.md"],
        rubric=(
            "Must state the employee is NOT currently eligible for remote work because they "
            "are still within the 90-day probation period (45 < 90 days), AND must state that "
            "they become eligible once the 90-day probation period ends. Both parts must be "
            "present and must not contradict each other."
        ),
        reference=(
            "The employee is not currently eligible for remote work because they are only 45 "
            "days into the 90-day probation period. They will become eligible once the 90-day "
            "probation period ends."
        ),
    ),
    dict(
        question=(
            "An employee started exactly 60 days before the addendum's 2025-07-01 effective "
            "date. Today they are 100 days into their job. Are they eligible for the new "
            "3-day remote work policy?"
        ),
        tag="regression: known gap - date disambiguation (documented, unresolved)",
        # FINDINGS.md §4 row 5: the model has been observed to conflate the addendum's
        # calendar effective date with the employee's personal tenure-based threshold date
        # when both numbers appear in the same context. Not yet fixed by chunking or model
        # size changes. known_issue=True means this case is tracked and reported separately
        # from the main pass rate, so it doesn't block eval.py on a known, documented gap --
        # but a regression run will still show it clearly if it gets worse, or flag it if a
        # future fix makes it pass so you remember to promote it out of "known gap" status.
        expected_keywords=["eligible", "100", "90"],
        expected_sources=["employee_handbook.md", "policy_addendum.md"],
        rubric=(
            "Must correctly separate two different dates: the addendum's calendar effective "
            "date (2025-07-01) and the employee's personal 90-day probation threshold (they "
            "are at 100 days of tenure, so past probation). Must conclude they ARE eligible "
            "for the 3-day remote work policy, based on tenure, not the addendum's effective date."
        ),
        reference=(
            "The employee is eligible for the 3-day remote work policy because they have 100 "
            "days of tenure, which is past the 90-day probation threshold. The addendum's "
            "2025-07-01 effective date is a separate calendar date and does not change this "
            "employee's personal eligibility, which is based on their own tenure."
        ),
        known_issue=True,
    ),
]


def score(answer, expected_any):
    """PASS if the answer contains any of the expected keywords (case-insensitive)."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_any)


def hit_rate(hits, expected_sources, k=3):
    """True if any expected source appears in the top-k retrieved chunks. None for out-of-scope."""
    if not expected_sources:
        return None
    top_k_sources = {h["source"] for h in hits[:k]}
    return any(src in top_k_sources for src in expected_sources)


def failure_label(hits, expected_sources, answer_ok, k=3):
    """Classify a result as OK, RETRIEVAL FAILURE, or GENERATION FAILURE."""
    if not expected_sources:
        return "out-of-scope"
    retrieved = hit_rate(hits, expected_sources, k)
    if not retrieved:
        return "RETRIEVAL FAILURE"
    return "OK" if answer_ok else "GENERATION FAILURE"


def run(store, collection_name, model, hybrid_retriever=None, top_k=8, use_judge=False):
    mode = "hybrid" if hybrid_retriever else "dense"
    print(f"\n{'='*80}\nCOLLECTION: {collection_name}  ({store.count(collection_name)} chunks)  mode={mode}\n{'='*80}")
    hit3_scores = []
    # per-tag tallies, excluding known_issue questions from the headline pass rate
    tag_stats = {}  # tag -> {"passed": int, "total": int}
    known_issue_results = []

    for q in QUESTIONS:
        question, tag = q["question"], q["tag"]
        expected_any, expected_sources = q["expected_keywords"], q["expected_sources"]
        rubric, is_known_issue = q.get("rubric"), q.get("known_issue", False)

        answer, hits = ask(
            store, collection_name, model, question,
            top_k=top_k, verbose=False, hybrid_retriever=hybrid_retriever,
        )
        answer_ok = score(answer, expected_any)
        h3 = hit_rate(hits, expected_sources, k=3)
        if h3 is not None:
            hit3_scores.append(h3)

        judge_line = ""
        if use_judge and rubric:
            from judge import judge as llm_judge
            result = llm_judge(question, rubric, answer)
            judge_line = f"  judge={result['score']}/5 ({result['reason']})" if result["score"] is not None \
                else f"  judge=ERR ({result['reason']})"

        label = failure_label(hits, expected_sources, answer_ok, k=3)
        sources_str = ", ".join(f"{h['source']}/{h['heading']}" for h in hits[:2])
        flag = " [KNOWN ISSUE — not counted in pass rate]" if is_known_issue else ""
        print(f"\n[{tag}]{flag} answer={'PASS' if answer_ok else 'FAIL'}  retrieval={'HIT' if h3 else ('MISS' if h3 is False else 'n/a')}  {label}{judge_line}")
        print(f"Q: {question}")
        print(f"Expected keywords: {expected_any}")
        print(f"Top sources: {sources_str}")
        print(f"A: {answer}")

        if is_known_issue:
            known_issue_results.append((tag, answer_ok))
        else:
            stats = tag_stats.setdefault(tag, {"passed": 0, "total": 0})
            stats["total"] += 1
            stats["passed"] += answer_ok

    hr3 = sum(hit3_scores) / len(hit3_scores) if hit3_scores else 0.0
    total_passed = sum(s["passed"] for s in tag_stats.values())
    total_count = sum(s["total"] for s in tag_stats.values())

    print(f"\n--- {collection_name} ({mode}) — by problem type ---")
    for tag, s in tag_stats.items():
        print(f"  {tag:<50} {s['passed']}/{s['total']}")
    if known_issue_results:
        print("  known issues (documented, excluded from pass rate above):")
        for tag, ok in known_issue_results:
            print(f"    {tag:<48} {'PASS (consider promoting out of known-issue status!)' if ok else 'still failing, as expected'}")

    print(f"\n--- {collection_name} ({mode}) — overall ---")
    print(f"Answer score : {total_passed}/{total_count}")
    print(f"hit-rate@3   : {hr3:.0%}  ({sum(hit3_scores)}/{len(hit3_scores)} in-scope questions)")
    return {"tag_stats": tag_stats, "passed": total_passed, "total": total_count, "hit_rate_3": hr3}


def compare_hitrate(store, collection_name, model, k=3, top_k=8):
    """Retrieval-only comparison: dense vs hybrid. No LLM calls — just measures whether
    the right document appears in the top-k. Proves the hybrid change helped (or not)."""
    from hybrid import HybridRetriever

    print(f"\n{'='*80}")
    print(f"BEFORE vs AFTER — hit-rate@{k} comparison  (retrieval only, no LLM)")
    print(f"Collection: {collection_name}  |  one change: dense → hybrid (BM25 + dense + RRF)")
    print(f"{'='*80}")
    print(f"\n{'Question':<52} {'Before':>8} {'After':>8} {'Change':>8}")
    print("-" * 80)

    hybrid = HybridRetriever(store, collection_name, model, QUERY_PREFIX)

    before_scores, after_scores = [], []
    for q in QUESTIONS:
        question, expected_sources = q["question"], q["expected_sources"]
        if not expected_sources:
            print(f"{question[:50]:<52} {'n/a':>8} {'n/a':>8} {'':>8}  (out-of-scope)")
            continue

        dense_hits = retrieve(store, collection_name, model, question, top_k=top_k, rerank=True)
        hybrid_hits = retrieve(store, collection_name, model, question, top_k=top_k, rerank=True, hybrid_retriever=hybrid)

        b = hit_rate(dense_hits, expected_sources, k)
        a = hit_rate(hybrid_hits, expected_sources, k)
        before_scores.append(b)
        after_scores.append(a)

        b_str = "HIT" if b else "MISS"
        a_str = "HIT" if a else "MISS"
        change = ("  same" if b == a else ("  +FIXED" if a and not b else "  -BROKE"))
        print(f"{question[:50]:<52} {b_str:>8} {a_str:>8} {change}")

    br = sum(before_scores) / len(before_scores) if before_scores else 0.0
    ar = sum(after_scores) / len(after_scores) if after_scores else 0.0
    print("-" * 80)
    print(f"{'OVERALL hit-rate@3':<52} {br:>7.0%} {ar:>7.0%} {'  +' + str(round((ar-br)*100)) + 'pp' if ar != br else '  no change'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compare", dest="compare", action="store_false", default=True,
                        help="skip the before/after hit-rate comparison")
    parser.add_argument("--judge", action="store_true",
                        help="also run the LLM judge on rubric-graded questions "
                             "(only trust this after judge_calibration.py passes)")
    parser.add_argument("--save", metavar="PATH",
                        help="save per-tag scores as JSON, for use with compare_eval_runs.py")
    args = parser.parse_args()

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    from hybrid import HybridRetriever
    hybrid = HybridRetriever(store, "hr_policy_parent_child", model, QUERY_PREFIX)

    run(store, "hr_policy_subsection", model, use_judge=args.judge)
    run(store, "hr_policy_section", model, use_judge=args.judge)
    result = run(store, "hr_policy_parent_child", model, hybrid_retriever=hybrid, use_judge=args.judge)

    if args.compare:
        compare_hitrate(store, "hr_policy_parent_child", model)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved hr_policy_parent_child results to {args.save}")

    print(f"\n{'='*80}\nDone.\n{'='*80}")
