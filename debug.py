"""
Inspection view: question → retrieved chunks → answer, side by side.

Shows exactly what the retriever fetched and why the answer succeeded or failed.
Use this to classify failures as RETRIEVAL FAILURE vs GENERATION FAILURE before
deciding what to fix.

Usage:
  python debug.py                                    # inspect all eval questions
  python debug.py --question "your question here"    # single question
  python debug.py --collection hr_policy_subsection  # compare a different collection
  python debug.py --no-hybrid                        # dense-only retrieval
"""
import argparse
from sentence_transformers import SentenceTransformer
from vectorstore import get_store
from query import (
    EMBED_MODEL_NAME, QUERY_PREFIX, CONFIDENCE_THRESHOLD, NO_MATCH_ANSWER,
    retrieve, build_prompt, call_openrouter,
)
from eval import QUESTIONS, score as eval_score, hit_rate, failure_label

PREVIEW_CHARS = 180


def inspect_one(store, collection_name, model, question,
                expected_any=None, expected_sources=None,
                top_k=8, hybrid_retriever=None, k_for_hitrate=3):

    print(f"\n{'='*72}")
    print(f"Q: {question}")
    print(f"{'='*72}")

    hits = retrieve(
        store, collection_name, model, question,
        top_k=top_k, rerank=True, hybrid_retriever=hybrid_retriever,
    )

    print(f"\nRetrieved chunks (showing top {min(len(hits), top_k)}):")
    for i, h in enumerate(hits[:top_k]):
        sim = f"sim={h['similarity']:.4f}"
        rrf = f"  rrf={h['rrf_score']:.4f}" if "rrf_score" in h else ""
        heading = h.get("parent_heading", h["heading"])
        in_expected = ("*" if expected_sources and h["source"] in expected_sources else " ")
        print(f"  [{i+1}]{in_expected} {h['source']} / {heading}  ({sim}{rrf})")
        preview = h["text"][:PREVIEW_CHARS].replace("\n", " ")
        print(f"       \"{preview}...\"")

    if expected_sources:
        print(f"\n  (* = expected source document)")

    best_sim = max(h["similarity"] for h in hits) if hits else 0.0
    if best_sim < CONFIDENCE_THRESHOLD:
        answer = NO_MATCH_ANSWER
        print(f"\n  [below confidence threshold {CONFIDENCE_THRESHOLD} — LLM skipped]")
    else:
        try:
            prompt = build_prompt(question, hits)
            answer = call_openrouter(prompt)
        except Exception as e:
            answer = f"[LLM error: {e}]"

    print(f"\nA: {answer[:700]}{'...' if len(answer) > 700 else ''}")

    # Failure diagnosis
    if expected_any is not None or expected_sources is not None:
        answer_ok = eval_score(answer, expected_any) if expected_any else True
        label = failure_label(hits, expected_sources or [], answer_ok, k=k_for_hitrate)
        h3 = hit_rate(hits, expected_sources or [], k=k_for_hitrate)

        print(f"\nDiagnosis      : {label}")
        print(f"Answer correct : {'YES' if answer_ok else 'NO'}  (expected any of {expected_any})")
        if expected_sources:
            print(f"hit-rate@{k_for_hitrate}     : {'HIT' if h3 else 'MISS'}  (expected source: {expected_sources})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="hr_policy_parent_child")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--question", default=None, help="single question to inspect")
    parser.add_argument("--hybrid", action="store_true", default=True)
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    args = parser.parse_args()

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    hybrid_retriever = None
    if args.hybrid:
        from hybrid import HybridRetriever
        hybrid_retriever = HybridRetriever(store, args.collection, model, QUERY_PREFIX)

    mode = "hybrid" if args.hybrid else "dense"
    print(f"\nInspection view — collection='{args.collection}'  mode={mode}  top_k={args.top_k}")

    if args.question:
        inspect_one(store, args.collection, model, args.question,
                    top_k=args.top_k, hybrid_retriever=hybrid_retriever)
    else:
        for question, tag, expected_any, expected_sources in QUESTIONS:
            inspect_one(
                store, args.collection, model, question,
                expected_any=expected_any, expected_sources=expected_sources,
                top_k=args.top_k, hybrid_retriever=hybrid_retriever,
            )
