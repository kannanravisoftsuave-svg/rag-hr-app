"""
Evaluation: automated scoring + failure classification + hit-rate@k.

Each question carries expected answer keywords (for answer correctness) and
expected source documents (for retrieval quality). Failures are classified as:
  RETRIEVAL FAILURE  — right document never appeared in top-k
  GENERATION FAILURE — right document was retrieved, but answer was wrong
  OK                 — retrieved correctly and answered correctly

Also measures hit-rate@3 (did the right document appear in the top 3 results?)
before (dense only) and after (hybrid BM25+dense) to prove the hybrid change helped.

Usage:
  python eval.py                  # full run: both collections + before/after compare
  python eval.py --no-compare     # skip the before/after comparison (faster)
"""
import argparse
from sentence_transformers import SentenceTransformer
from vectorstore import get_store
from query import ask, retrieve, EMBED_MODEL_NAME, QUERY_PREFIX

# (question, tag, expected_keywords, expected_sources)
# expected_keywords: PASS if ANY keyword appears in the answer (case-insensitive)
# expected_sources:  source filenames that MUST appear in top-k for retrieval to count as a hit
#                    empty list = out-of-scope question (no retrieval check)
QUESTIONS = [
    (
        "How many days of paid sick leave do employees get per year?",
        "basic fact",
        ["10 days", "10 paid days"],
        ["employee_handbook.md"],
    ),
    (
        "How many weeks of paid parental leave does a primary caregiver get?",
        "addendum supersedes handbook",
        ["16 weeks"],
        ["policy_addendum.md"],
    ),
    (
        "What is the home-office equipment allowance?",
        "addendum supersedes handbook",
        ["$500", "500"],
        ["policy_addendum.md"],
    ),
    (
        "How many days per week can eligible employees work remotely?",
        "addendum supersedes handbook",
        ["3 days", "three days"],
        ["policy_addendum.md"],
    ),
    (
        "What is the probation period for new hires?",
        "basic fact, handbook only",
        ["90-day", "90 day", "90 days"],
        ["employee_handbook.md"],
    ),
    (
        "Can an employee paste customer data into an unapproved AI tool?",
        "new addendum-only policy",
        ["no", "not", "must never", "confidential"],
        ["policy_addendum.md"],
    ),
    (
        "What is the company's policy on stock options for employees?",
        "NOT in documents -> should refuse",
        ["don't know", "not mentioned", "no information", "does not"],
        [],  # out-of-scope: no expected source
    ),
    (
        "What is the annual performance bonus cap?",
        "basic fact, handbook only",
        ["10%"],
        ["employee_handbook.md"],
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


def run(store, collection_name, model, hybrid_retriever=None, top_k=8):
    mode = "hybrid" if hybrid_retriever else "dense"
    print(f"\n{'='*80}\nCOLLECTION: {collection_name}  ({store.count(collection_name)} chunks)  mode={mode}\n{'='*80}")
    passed = 0
    hit3_scores = []

    for question, tag, expected_any, expected_sources in QUESTIONS:
        answer, hits = ask(
            store, collection_name, model, question,
            top_k=top_k, verbose=False, hybrid_retriever=hybrid_retriever,
        )
        answer_ok = score(answer, expected_any)
        passed += answer_ok
        h3 = hit_rate(hits, expected_sources, k=3)
        if h3 is not None:
            hit3_scores.append(h3)

        label = failure_label(hits, expected_sources, answer_ok, k=3)
        sources_str = ", ".join(f"{h['source']}/{h['heading']}" for h in hits[:2])
        print(f"\n[{tag}] answer={'PASS' if answer_ok else 'FAIL'}  retrieval={'HIT' if h3 else ('MISS' if h3 is False else 'n/a')}  {label}")
        print(f"Q: {question}")
        print(f"Expected keywords: {expected_any}")
        print(f"Top sources: {sources_str}")
        print(f"A: {answer}")

    hr3 = sum(hit3_scores) / len(hit3_scores) if hit3_scores else 0.0
    print(f"\n--- {collection_name} ({mode}) ---")
    print(f"Answer score : {passed}/{len(QUESTIONS)}")
    print(f"hit-rate@3   : {hr3:.0%}  ({sum(hit3_scores)}/{len(hit3_scores)} in-scope questions)")
    return passed, hr3


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
    for question, tag, _, expected_sources in QUESTIONS:
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
    args = parser.parse_args()

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    from hybrid import HybridRetriever
    hybrid = HybridRetriever(store, "hr_policy_parent_child", model, QUERY_PREFIX)

    run(store, "hr_policy_subsection", model)
    run(store, "hr_policy_section", model)
    run(store, "hr_policy_parent_child", model, hybrid_retriever=hybrid)

    if args.compare:
        compare_hitrate(store, "hr_policy_parent_child", model)

    print(f"\n{'='*80}\nDone.\n{'='*80}")
