"""
Structured LLM output for HR policy answers (Week 2 upgrade, applied retroactively).

query.py's original SYSTEM_PROMPT asks the model to write free-text "Quotes:" /
"Reasoning:" / "Answer:" sections. Nothing validates the model actually followed
that format -- if it skips a section, reorders them, or contradicts itself
between Reasoning and Answer (the exact bug in FINDINGS.md #4), the app has no
way to detect it. It just ships whatever text came back.

This module replaces free-text parsing with a Pydantic-validated JSON contract:
if the model's output doesn't match HRAnswer's schema, we retry with the
specific validation error fed back to it (the "validation & retry" loop Week 2
asked for), instead of shipping a malformed answer.

This is ADDITIVE, not a replacement -- query.ask() and the original SYSTEM_PROMPT
are untouched, so eval.py, judge.py, ragas_eval.py, and the existing API/UI paths
keep working exactly as before. Use ask_structured() where you want the new,
validated path.
"""
from __future__ import annotations
import json

from pydantic import BaseModel, Field, ValidationError

from query import call_openrouter, retrieve, CONFIDENCE_THRESHOLD, NO_MATCH_ANSWER


class HRAnswer(BaseModel):
    quotes: list[str] = Field(
        description="Exact sentences copied word-for-word from the context that support the "
                    "answer. Empty list only if is_refusal is true."
    )
    reasoning: str = Field(
        description="Numbers, dates, or conditions extracted from the quotes, explicitly applied "
                    "to the specific situation in the question."
    )
    answer: str = Field(
        description="The final answer to give the user. Must be consistent with reasoning -- "
                    "never contradict it."
    )
    sources: list[str] = Field(
        description='Citations in the form "<file>, Section: <heading>", one per claim. '
                    'Empty list only if is_refusal is true.'
    )
    is_refusal: bool = Field(
        description="True ONLY if the context genuinely does not contain the answer."
    )


STRUCTURED_SYSTEM_PROMPT = """You are an HR policy assistant. Answer the user's question using ONLY the
context provided below.

Follow these rules:
1. quotes: copy the exact sentence(s) from the context most relevant to the question, word for word,
   with no paraphrasing.
2. reasoning: extract every specific number, date, or condition from the quotes that is relevant to the
   question, and explicitly apply it to the situation described (e.g. compare a stated tenure/date
   against any threshold found in the context). If the question has multiple parts, resolve each part
   independently -- never let one unanswerable part erase a part you already correctly worked out.
3. The absence of a stated restriction is NOT the same as an explicit permission, and the absence of a
   stated permission is NOT the same as a prohibition. Only answer "yes" or "no" if the context
   explicitly states that answer.
4. answer: the final answer, consistent with your reasoning above -- if reasoning reached a conclusion,
   state that conclusion; never contradict it.
5. sources: cite the source document and section heading for every claim, e.g.
   "employee_handbook.md, Section: Sick Leave". If multiple chunks conflict (e.g. an addendum updates a
   base policy), prefer the most recent/effective one and say so in the answer.
6. is_refusal: true ONLY if the context genuinely does not contain the answer for this part of the
   question. Do not guess, do not use outside knowledge, do not invent a citation.

Respond with ONLY a JSON object matching this schema, no other text, no markdown code fences:
{schema}

Context:
{context}

Question: {question}
"""


def build_structured_prompt(question: str, hits: list) -> str:
    """Same context-block construction as query.build_prompt() (dedupe by parent section, prefer
    parent_text when available) -- kept in sync manually since the two prompts differ in framing."""
    context_blocks = []
    seen = set()
    for h in hits:
        context_text = h.get("parent_text", h["text"])
        heading = h.get("parent_heading", h["heading"])
        key = (h["source"], heading)
        if key in seen:
            continue
        seen.add(key)
        context_blocks.append(f"[Source: {h['source']} | Section: {heading}]\n{context_text}")
    context = "\n\n---\n\n".join(context_blocks)
    schema = json.dumps(HRAnswer.model_json_schema(), indent=2)
    return STRUCTURED_SYSTEM_PROMPT.format(schema=schema, context=context, question=question)


def parse_structured_answer(raw: str) -> HRAnswer:
    """Extract and validate the model's JSON response. Raises ValueError with a message suitable
    for feeding straight back to the model on retry."""
    text = raw.strip()
    # Some free-tier models wrap JSON in ```json ... ``` fences despite instructions not to.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            text = rest if first_line.strip().lower() in ("json", "") else text
    try:
        return HRAnswer.model_validate_json(text)
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"Response did not match the required schema: {exc}") from exc


def ask_structured(store, collection_name, model, question, top_k=8, hybrid_retriever=None,
                    confidence_threshold=None, max_retries=2, rerank=True) -> HRAnswer:
    """Same retrieval path as query.ask(), but returns a validated HRAnswer instead of raw text.
    On invalid JSON / schema mismatch, retries with the validation error appended to the prompt --
    this is the "validation & retry" behavior the free-text path never had."""
    threshold = CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
    hits = retrieve(store, collection_name, model, question, top_k=top_k, rerank=rerank,
                     hybrid_retriever=hybrid_retriever)

    best_similarity = max((h["similarity"] for h in hits), default=0.0)
    if best_similarity < threshold:
        return HRAnswer(
            quotes=[], reasoning="Best retrieval similarity below confidence threshold.",
            answer=NO_MATCH_ANSWER, sources=[], is_refusal=True,
        )

    prompt = build_structured_prompt(question, hits)
    last_error = None
    for attempt in range(max_retries + 1):
        raw = call_openrouter(prompt, response_format={"type": "json_object"})
        try:
            return parse_structured_answer(raw)
        except ValueError as exc:
            last_error = exc
            prompt += f"\n\nYour previous response was invalid: {exc}\nReturn ONLY the corrected JSON object."

    raise RuntimeError(
        f"Model failed to return valid structured output after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
