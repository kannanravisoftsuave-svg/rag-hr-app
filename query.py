"""
Ask questions against the ingested HR policy documents.

Generation runs via OpenRouter (hosted). Requires an OPENROUTER_API_KEY
environment variable — get a free key at https://openrouter.ai/keys.

Usage:
  python query.py                          # interactive CLI, uses "section" chunking
  python query.py --collection hr_policy_subsection
  python query.py --model meta-llama/llama-3.1-8b-instruct:free
"""
import argparse
import os
import requests
from sentence_transformers import SentenceTransformer
from vectorstore import get_store

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Below this best-match similarity, retrieval almost certainly didn't find a real answer.
# Calibrated empirically on this corpus (10-question sample, corrected cosine similarity
# after fixing the Chroma L2-vs-cosine bug — see FINDINGS.md): correct top-hits clustered
# 0.78-0.90, genuinely unanswerable questions clustered 0.55-0.65. 0.70 sits in that gap.
# Small sample — re-tune with more questions via eval.py if this misfires on your corpus.
CONFIDENCE_THRESHOLD = 0.70
NO_MATCH_ANSWER = "I don't know based on the provided documents. (no confident match found in retrieval)"

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


def retrieve(store, collection_name, model, question, top_k=4):
    query_vec = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()[0]
    return store.query(collection_name, query_vec, top_k)


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


def call_openrouter(prompt):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a free key at https://openrouter.ai/keys, then set it, e.g.:\n"
            "  cmd:        set OPENROUTER_API_KEY=sk-or-...\n"
            "  PowerShell: $env:OPENROUTER_API_KEY = 'sk-or-...'"
        )
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def ask(store, collection_name, model, question, top_k=4, verbose=True, confidence_threshold=CONFIDENCE_THRESHOLD):
    hits = retrieve(store, collection_name, model, question, top_k=top_k)
    if verbose:
        print("\nRetrieved chunks:")
        for h in hits:
            print(f"  - [{h['source']} / {h['heading']}] (similarity={h['similarity']:.4f})")

    best_similarity = max((h["similarity"] for h in hits), default=0.0)
    if best_similarity < confidence_threshold:
        if verbose:
            print(f"  Best similarity {best_similarity:.4f} < threshold {confidence_threshold} — skipping LLM call.")
        return NO_MATCH_ANSWER, hits

    prompt = build_prompt(question, hits)
    answer = call_openrouter(prompt)
    return answer, hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="hr_policy_section")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="similarity below this skips the LLM call and refuses immediately; use 0 to disable")
    args = parser.parse_args()
    OPENROUTER_MODEL = args.model

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Using collection '{args.collection}' ({store.count(args.collection)} chunks). Model: {OPENROUTER_MODEL}. Confidence threshold: {args.threshold}")
    print("Type a question, or 'exit' to quit.\n")

    while True:
        question = input("Q> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer, hits = ask(store, args.collection, model, question, top_k=args.top_k, confidence_threshold=args.threshold)
        print(f"\nA> {answer}\n")
