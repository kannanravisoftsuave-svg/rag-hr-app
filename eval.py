"""
Chunk-size comparison: run the same question set against both the
'hr_policy_subsection' (small chunks) and 'hr_policy_section' (larger chunks)
collections, and print a side-by-side result so you can see how chunking
strategy affects retrieval/answer quality.

Usage: python eval.py
"""
import chromadb
from sentence_transformers import SentenceTransformer
from query import ask

DB_DIR = "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Each question is tagged with what it's testing.
QUESTIONS = [
    ("How many days of paid sick leave do employees get per year?", "basic fact"),
    ("How many weeks of paid parental leave does a primary caregiver get?", "addendum supersedes handbook"),
    ("What is the home-office equipment allowance?", "addendum supersedes handbook"),
    ("How many days per week can eligible employees work remotely?", "addendum supersedes handbook"),
    ("What is the probation period for new hires?", "basic fact, handbook only"),
    ("Can an employee paste customer data into an unapproved AI tool?", "new addendum-only policy"),
    ("What is the company's policy on stock options for employees?", "NOT in documents -> should refuse"),
    ("What is the annual performance bonus cap?", "basic fact, handbook only"),
]


def run(collection_name):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(collection_name)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"\n{'='*80}\nCOLLECTION: {collection_name}  ({collection.count()} chunks)\n{'='*80}")
    for question, tag in QUESTIONS:
        answer, hits = ask(collection, model, question, top_k=4, verbose=False)
        sources = ", ".join(f"{h['source']}/{h['heading']}" for h in hits[:2])
        print(f"\n[{tag}]")
        print(f"Q: {question}")
        print(f"Top sources: {sources}")
        print(f"A: {answer}")


if __name__ == "__main__":
    run("hr_policy_subsection")
    run("hr_policy_section")
