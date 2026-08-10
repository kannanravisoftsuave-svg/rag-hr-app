"""
Ask questions against the ingested HR policy documents.

Usage:
  python query.py                          # interactive CLI, uses "section" chunking
  python query.py --collection hr_policy_subsection
"""
import argparse
import json
import chromadb
import requests
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# BGE models expect this instruction prefix on the query side for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = """You are an HR policy assistant. Answer the user's question using ONLY the
context provided below. Follow these rules strictly:

1. First, in a section called "Quotes:", copy the exact sentence(s) from the context that are
   most relevant to the question, word for word, with no paraphrasing. Do this even if the
   question seems simple. If the question asks about a specific item (a person, category, date),
   check whether that exact item is named anywhere in the context before concluding it isn't.
2. Then, in a section called "Reasoning:", extract every specific number, date, or condition
   from the Quotes above that is relevant to the question, and explicitly apply it to the
   specific situation described in the question (e.g. compare the employee's stated tenure/date
   against any threshold found in the context). Base this only on what the Quotes actually say.
3. If the question has multiple parts, treat each part independently. Answer every part the
   context supports, even if another part is unanswerable. Never let one unanswerable part
   cause you to discard a part you already correctly worked out in your Reasoning.
4. The absence of a stated restriction is NOT the same as an explicit permission, and the
   absence of a stated permission is NOT the same as a prohibition. Only answer "yes" or "no"
   to a question if the context explicitly states that answer. If the context is simply silent
   on the specific scenario asked about (even if it covers the general topic), say so plainly
   instead of inferring an answer from what isn't mentioned.
5. Then, in a section called "Answer:", give the final answer clearly, citing the source
   document and section heading for every claim, in the format (Source: <file>, Section: <heading>).
   The Answer must be consistent with your Reasoning above — if your Reasoning reached a
   conclusion, state that conclusion in the Answer, do not contradict it.
6. Only say "I don't know based on the provided documents" for a part of the question that your
   Reasoning genuinely could not resolve from the context. Do not guess, do not use outside
   knowledge, do not make up a citation.
7. If multiple context chunks are relevant (e.g. an addendum updates a base policy),
   prefer the most recent/effective one and mention that it supersedes the earlier one.
"""


def retrieve(collection, model, question, top_k=4):
    query_vec = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_k)
    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "heading": meta["heading"], "distance": dist})
    return hits


def build_prompt(question, hits):
    context_blocks = []
    for h in hits:
        context_blocks.append(f"[Source: {h['source']} | Section: {h['heading']}]\n{h['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""


def call_ollama(prompt):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ask(collection, model, question, top_k=4, verbose=True):
    hits = retrieve(collection, model, question, top_k=top_k)
    prompt = build_prompt(question, hits)
    answer = call_ollama(prompt)
    if verbose:
        print("\nRetrieved chunks:")
        for h in hits:
            print(f"  - [{h['source']} / {h['heading']}] (distance={h['distance']:.4f})")
    return answer, hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="hr_policy_section")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()
    OLLAMA_MODEL = args.model

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(args.collection)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Using collection '{args.collection}' ({collection.count()} chunks). Model: {OLLAMA_MODEL}")
    print("Type a question, or 'exit' to quit.\n")

    while True:
        question = input("Q> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer, hits = ask(collection, model, question, top_k=args.top_k)
        print(f"\nA> {answer}\n")
