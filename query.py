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
import time
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from vectorstore import get_store

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading cross-encoder '{CROSS_ENCODER_MODEL}'...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"
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


def retrieve(store, collection_name, model, question, top_k=8, rerank=True, hybrid_retriever=None):
    candidate_k = top_k * 3 if rerank else top_k
    if hybrid_retriever is not None:
        hits = hybrid_retriever.retrieve(question, top_k=candidate_k)
    else:
        query_vec = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()[0]
        hits = store.query(collection_name, query_vec, candidate_k)
    if rerank and len(hits) > top_k:
        ce = get_cross_encoder()
        scores = ce.predict([(question, h["text"]) for h in hits])
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)][:top_k]
    return hits


def build_prompt(question, hits):
    context_blocks = []
    seen = set()
    for h in hits:
        # For parent-child collections, send the full parent section to the LLM; fall back to
        # the child text for collections that don't carry parent_text. Deduplicate by content
        # so multiple children of the same parent don't bloat the context with repeated text.
        context_text = h.get("parent_text", h["text"])
        heading = h.get("parent_heading", h["heading"])
        key = (h["source"], heading)
        if key in seen:
            continue
        seen.add(key)
        context_blocks.append(f"[Source: {h['source']} | Section: {heading}]\n{context_text}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""


def call_openrouter(prompt, max_retries=4):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a free key at https://openrouter.ai/keys, then set it, e.g.:\n"
            "  cmd:        set OPENROUTER_API_KEY=sk-or-...\n"
            "  PowerShell: $env:OPENROUTER_API_KEY = 'sk-or-...'"
        )
    for attempt in range(max_retries):
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
        if resp.status_code == 429:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
            print(f"  [rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    resp.raise_for_status()  # raise after exhausting retries


def ask(store, collection_name, model, question, top_k=8, verbose=True, confidence_threshold=CONFIDENCE_THRESHOLD, rerank=True, hybrid_retriever=None):
    hits = retrieve(store, collection_name, model, question, top_k=top_k, rerank=rerank, hybrid_retriever=hybrid_retriever)
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
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="similarity below this skips the LLM call and refuses immediately; use 0 to disable")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="disable cross-encoder re-ranking (faster but lower precision)")
    parser.add_argument("--hybrid", action="store_true", default=True, help="use BM25+dense hybrid search (default on)")
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false", help="disable hybrid search, use dense only")
    args = parser.parse_args()
    OPENROUTER_MODEL = args.model

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    hybrid_retriever = None
    if args.hybrid:
        from hybrid import HybridRetriever
        hybrid_retriever = HybridRetriever(store, args.collection, model, QUERY_PREFIX)

    print(f"Using collection '{args.collection}' ({store.count(args.collection)} chunks). Model: {OPENROUTER_MODEL}. Threshold: {args.threshold}. Re-rank: {args.rerank}. Hybrid: {args.hybrid}")
    print("Type a question, or 'exit' to quit.\n")

    while True:
        question = input("Q> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer, hits = ask(store, args.collection, model, question, top_k=args.top_k, confidence_threshold=args.threshold, rerank=args.rerank, hybrid_retriever=hybrid_retriever)
        print(f"\nA> {answer}\n")
