"""
Chunk-size comparison + automated scoring: run the same question set against
both the 'hr_policy_subsection' (small chunks) and 'hr_policy_section'
(larger chunks) collections, and automatically score each answer against
expected keywords instead of requiring a human to read every answer.

Usage: python eval.py
"""
from sentence_transformers import SentenceTransformer
from vectorstore import get_store
from query import ask

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Each question: (question, tag, expected_any) — PASS if the answer contains
# ANY one of the expected_any substrings (case-insensitive). This is deliberately
# loose (OR, not AND) since free-text LLM answers vary in phrasing — it's checking
# "did the key fact make it into the answer," not exact wording.
QUESTIONS = [
    ("How many days of paid sick leave do employees get per year?",
     "basic fact", ["10 days", "10 paid days"]),
    ("How many weeks of paid parental leave does a primary caregiver get?",
     "addendum supersedes handbook", ["16 weeks"]),
    ("What is the home-office equipment allowance?",
     "addendum supersedes handbook", ["$500", "500"]),
    ("How many days per week can eligible employees work remotely?",
     "addendum supersedes handbook", ["3 days", "three days"]),
    ("What is the probation period for new hires?",
     "basic fact, handbook only", ["90-day", "90 day", "90 days"]),
    ("Can an employee paste customer data into an unapproved AI tool?",
     "new addendum-only policy", ["no", "not", "must never", "confidential"]),
    ("What is the company's policy on stock options for employees?",
     "NOT in documents -> should refuse", ["don't know", "not mentioned", "no information", "does not"]),
    ("What is the annual performance bonus cap?",
     "basic fact, handbook only", ["10%"]),
]


def score(answer, expected_any):
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_any)


def run(store, collection_name, model):
    print(f"\n{'='*80}\nCOLLECTION: {collection_name}  ({store.count(collection_name)} chunks)\n{'='*80}")
    passed = 0
    for question, tag, expected_any in QUESTIONS:
        answer, hits = ask(store, collection_name, model, question, top_k=4, verbose=False)
        sources = ", ".join(f"{h['source']}/{h['heading']}" for h in hits[:2])
        ok = score(answer, expected_any)
        passed += ok
        print(f"\n[{tag}] {'PASS' if ok else 'FAIL'}")
        print(f"Q: {question}")
        print(f"Expected any of: {expected_any}")
        print(f"Top sources: {sources}")
        print(f"A: {answer}")
    print(f"\n--- Score for '{collection_name}': {passed}/{len(QUESTIONS)} ---")
    return passed


if __name__ == "__main__":
    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    sub_score = run(store, "hr_policy_subsection", model)
    sec_score = run(store, "hr_policy_section", model)

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"hr_policy_subsection: {sub_score}/{len(QUESTIONS)}")
    print(f"hr_policy_section:    {sec_score}/{len(QUESTIONS)}")
