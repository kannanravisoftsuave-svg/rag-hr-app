"""
LLM-as-judge: grades an answer's factual correctness against a rubric, for
cases a keyword match can't score well (multi-part synthesis, "did it reach
the right conclusion" rather than "does this exact phrase appear").

Uses the same OpenRouter backend as generation (see query.py). Do NOT trust
this score until you've run judge_calibration.py and confirmed it agrees with
your own grading on a sample -- an unvalidated judge is just a confident
number nobody should rely on.
"""
import json
import re

from query import call_openrouter

JUDGE_PROMPT = """You are grading an HR policy assistant's answer for factual correctness ONLY.
Ignore style, tone, and citation formatting -- only check whether the answer reaches the
right conclusion and supports it with the right facts.

Question: {question}

Rubric -- what a correct answer must say:
{rubric}

Assistant's answer:
{answer}

Score the answer's correctness against the rubric on a 1-5 scale:
5 = fully correct, matches the rubric
4 = correct conclusion, missing a minor supporting detail
3 = partially correct -- gets some of the rubric right, misses or garbles part of it
2 = mostly wrong but touches the right topic
1 = wrong conclusion, refuses when the rubric says it shouldn't, or answers with no basis in the rubric

Reply with ONLY a JSON object, no other text: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge(question, rubric, answer):
    """Returns {"score": 1-5 or None, "reason": str}. score is None if the judge's
    output couldn't be parsed -- treat that as a judge failure, not a score of 0."""
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric, answer=answer)
    raw = call_openrouter(prompt)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"score": None, "reason": f"judge returned unparseable output: {raw[:200]}"}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {"score": None, "reason": f"judge returned invalid JSON: {raw[:200]}"}
    return {"score": parsed.get("score"), "reason": parsed.get("reason", "")}
