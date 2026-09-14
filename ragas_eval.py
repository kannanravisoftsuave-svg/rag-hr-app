"""
RAGAS-based evaluation -- an alternative to the hand-rolled judge.py, using
RAGAS's pre-built, pre-tested LLM-judge metrics instead of a rubric you wrote
and validated yourself.

Reuses the same QUESTIONS list as eval.py (see that file for the "why" of each
question) so there's one answer key, not two. Runs the SAME questions through
the SAME app, but scores them with four RAGAS metrics instead of keyword match:

  faithfulness       Is the answer grounded in the retrieved text, or did it
                      make something up? (hallucination check)
  answer_relevancy   Does the answer actually address the question asked?
  context_precision  Of the chunks retrieved, how many were actually useful?
  context_recall     Did retrieval find everything needed to answer correctly?

context_precision and context_recall need a ground-truth "reference" answer to
compare against -- that's why QUESTIONS entries in eval.py carry a `reference`
field.

Every RAGAS metric here is itself computed by an LLM call (RAGAS ships the
grading prompts; it still needs a model to run them) -- routed through the
same OpenRouter account as generation, via a LangChain-compatible wrapper.
Answer_relevancy also needs an embedding model to compare meanings; this uses
the SAME local BGE model as retrieval, so no extra API cost there.

Setup: see EVALS_SETUP.md for the one-time `pip install` this needs and why.

Usage:
  python ragas_eval.py                    # full run, all questions, all 4 metrics
  python ragas_eval.py --save ragas.json  # save per-tag averages for comparison
"""
import argparse
import json
import os

from sentence_transformers import SentenceTransformer
from vectorstore import get_store
from query import ask, EMBED_MODEL_NAME, OPENROUTER_MODEL, QUERY_PREFIX
from hybrid import HybridRetriever
from eval import QUESTIONS

COLLECTION = "hr_policy_parent_child"


def context_texts_for(hit_list):
    """Recreate the same de-duplicated context blocks build_prompt() in query.py sends to
    the LLM, so RAGAS is judging faithfulness against what the model actually saw."""
    seen, texts = set(), []
    for h in hit_list:
        text = h.get("parent_text", h["text"])
        key = (h["source"], h.get("parent_heading", h["heading"]))
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return texts


def build_dataset(store, model, hybrid_retriever, top_k=8):
    """Ask every question for real, and assemble RAGAS's expected sample shape:
    user_input (question), response (answer), retrieved_contexts (list of chunk
    text), reference (ground truth). Returns (samples, tag_by_index) so results
    can be grouped by problem type afterwards."""
    from query import retrieve

    samples, tags = [], []
    for q in QUESTIONS:
        print(f"Asking: {q['question']}")
        answer, hits = ask(
            store, COLLECTION, model, q["question"],
            top_k=top_k, verbose=False, hybrid_retriever=hybrid_retriever,
        )
        samples.append({
            "user_input": q["question"],
            "response": answer,
            "retrieved_contexts": context_texts_for(hits) or [""],  # RAGAS needs a non-empty list
            "reference": q.get("reference", ""),
        })
        tags.append(q["tag"])
    return samples, tags


def run(top_k=8):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set -- see EVALS_SETUP.md")

    # RAGAS wants a LangChain chat model and embeddings model, not raw API calls.
    # OpenRouter speaks the OpenAI chat-completions format, so ChatOpenAI works
    # against it directly by pointing base_url at OpenRouter instead of OpenAI.
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME))

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    store = get_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    hybrid = HybridRetriever(store, COLLECTION, model, QUERY_PREFIX)

    samples, tags = build_dataset(store, model, hybrid, top_k=top_k)
    dataset = EvaluationDataset.from_list(samples)

    print(f"\nRunning RAGAS ({len(metrics)} metrics x {len(samples)} questions -- this makes real LLM calls)...")
    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()
    df["tag"] = tags

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    print(f"\n{'='*100}\nPer-question scores\n{'='*100}")
    for _, row in df.iterrows():
        scores = "  ".join(f"{c}={row[c]:.2f}" for c in metric_cols if row[c] is not None)
        print(f"[{row['tag']}] {row['user_input'][:60]}\n    {scores}")

    print(f"\n{'='*100}\nAverage by problem type (tag)\n{'='*100}")
    by_tag = df.groupby("tag")[metric_cols].mean()
    print(by_tag.round(2).to_string())

    print(f"\n{'='*100}\nOverall averages\n{'='*100}")
    overall = df[metric_cols].mean().round(3).to_dict()
    for metric, value in overall.items():
        print(f"  {metric:<20} {value}")

    return {"by_tag": by_tag.round(3).to_dict(orient="index"), "overall": overall}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="save per-tag + overall averages as JSON")
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    result = run(top_k=args.top_k)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved RAGAS results to {args.save}")
