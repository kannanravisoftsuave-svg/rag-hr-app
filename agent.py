"""
Week 7 — a hand-built ReAct agent loop for the HR policy assistant.

Unlike query.ask() / structured_answer.ask_structured() (both: retrieve once,
generate once, done), this loops: think -> act -> observe -> repeat, until the
model calls the `finish` tool or a safety limit trips. The model decides how
many times to search and whether to reach for the date-math tool -- nothing
here is hardcoded into a fixed sequence.

Why calculate_tenure exists: FINDINGS.md #4 documents a real, still-open bug
(known_issue=True in eval.py) where the model conflates two different dates
when doing tenure math in its head. Giving it a tool that computes the exact
answer, instead of asking it to do arithmetic in free text, is a real fix
attempt for that bug -- not a toy exercise.

No framework (no LangChain/LangGraph) -- the loop is ~90 lines below so it's
never a black box. Every step is printed as it happens (the "visible steps"
requirement from the brief).

Usage:
  python agent.py "A new employee is 45 days into their job..."
"""
from __future__ import annotations
import json
import time
from datetime import date
from typing import Literal

from pydantic import BaseModel, ValidationError

from query import call_openrouter, retrieve, EMBED_MODEL_NAME, QUERY_PREFIX
from vectorstore import get_store

MAX_STEPS = 6
MAX_SECONDS = 90
COLLECTION = "hr_policy_parent_child"

TOOL_DESCRIPTIONS = """You have exactly three tools:

1. search_policy_docs -- action_input: {"query": "<search text>"}
   Searches the HR policy documents (handbook + addendum) and returns the top matching
   passages with their source file and section heading. Use this to find facts and dates.

2. calculate_tenure -- action_input: {"start_date": "YYYY-MM-DD", "as_of_date": "YYYY-MM-DD"}
   Computes the exact number of days between two dates. ALWAYS use this tool for any date
   arithmetic (comparing a start date to a threshold date, checking if N days have passed) --
   never compute date math yourself in the Thought field. You are unreliable at mental date
   arithmetic; this tool is not.

3. finish -- action_input: {"answer": "<final answer text, with citations>"}
   Call this when you have everything needed to answer. This ends the loop.
"""

AGENT_SYSTEM_PROMPT = """You are an HR policy assistant that works in a loop: Thought, Action,
Observation, repeated until you call finish.

{tools}

At each step, respond with ONLY a JSON object of this exact shape, no other text:
{{"thought": "<your reasoning about what to do next>", "action": "<tool name>", "action_input": {{...}}}}

Rules:
- One action per step. Do not skip straight to an answer without using search_policy_docs at
  least once, unless the question needs no policy lookup at all.
- If a question has multiple parts (e.g. "is this allowed, and if not, when"), make sure your
  Thought plans for resolving every part before you call finish.
- Never guess a date, number, or policy detail that search_policy_docs hasn't shown you.
- When your reasoning involves comparing dates or counting days, call calculate_tenure -- do
  not do it in your head.

Question: {question}

So far:
{transcript}

What is your next step? Respond with ONLY the JSON object described above."""


class AgentStep(BaseModel):
    thought: str
    action: Literal["search_policy_docs", "calculate_tenure", "finish"]
    action_input: dict


def parse_step(raw: str) -> AgentStep:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            text = rest if first_line.strip().lower() in ("json", "") else text
    try:
        return AgentStep.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"Response did not match the required step schema: {exc}") from exc


def tool_search_policy_docs(store, embed_model, hybrid_retriever, action_input: dict) -> str:
    query = action_input.get("query")
    if not query:
        return 'ERROR: action_input must include "query", e.g. {"query": "remote work eligibility"}'
    hits = retrieve(store, COLLECTION, embed_model, query, top_k=5, hybrid_retriever=hybrid_retriever)
    if not hits:
        return "No matching passages found."
    lines = []
    for h in hits[:5]:
        text = h.get("parent_text", h["text"])[:500]
        lines.append(f"[{h['source']} | Section: {h.get('parent_heading', h['heading'])}]\n{text}")
    return "\n\n---\n\n".join(lines)


def tool_calculate_tenure(action_input: dict) -> str:
    start = action_input.get("start_date")
    as_of = action_input.get("as_of_date")
    if not start or not as_of:
        return 'ERROR: action_input must include both "start_date" and "as_of_date" as YYYY-MM-DD.'
    try:
        start_d = date.fromisoformat(start)
        as_of_d = date.fromisoformat(as_of)
    except ValueError as exc:
        return f"ERROR: could not parse dates ({exc}). Use YYYY-MM-DD format."
    days = (as_of_d - start_d).days
    return f"{days} days between {start} and {as_of} (as_of_date is {'after' if days >= 0 else 'before'} start_date)."


def run_agent(question: str, max_steps: int = MAX_STEPS, max_seconds: int = MAX_SECONDS) -> dict:
    """Returns {"answer", "transcript", "num_llm_calls", "num_tool_calls", "elapsed_seconds",
    "stopped_reason", "num_steps"}."""
    store = get_store()
    from sentence_transformers import SentenceTransformer
    from hybrid import HybridRetriever
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    hybrid_retriever = HybridRetriever(store, COLLECTION, embed_model, QUERY_PREFIX)

    transcript_lines: list[str] = []
    last_action_signature = None
    repeat_streak = 0
    num_llm_calls = 0
    num_tool_calls = 0
    start_time = time.time()
    answer = None
    stopped_reason = "max_steps"

    for step_num in range(1, max_steps + 1):
        if time.time() - start_time > max_seconds:
            stopped_reason = "timeout"
            break

        prompt = AGENT_SYSTEM_PROMPT.format(
            tools=TOOL_DESCRIPTIONS, question=question,
            transcript="\n".join(transcript_lines) or "(nothing yet -- this is the first step)",
        )

        raw = call_openrouter(prompt, response_format={"type": "json_object"})
        num_llm_calls += 1
        try:
            step = parse_step(raw)
        except ValueError as exc:
            # one retry with the parse error fed back, same pattern as structured_answer.py
            retry_prompt = prompt + f"\n\nYour previous response was invalid: {exc}\nReturn ONLY the corrected JSON object."
            raw = call_openrouter(retry_prompt, response_format={"type": "json_object"})
            num_llm_calls += 1
            step = parse_step(raw)  # let a second failure raise -- caller sees it, no silent bad state

        print(f"\n[step {step_num}] Thought: {step.thought}")
        print(f"[step {step_num}] Action: {step.action}  Input: {step.action_input}")

        # Only flags CONSECUTIVE identical actions as "stuck" -- a legitimate non-consecutive
        # repeat (e.g. re-checking the same search two steps later for a good reason) must not
        # trip this, or the agent gets unfairly penalized in the race for doing something sane.
        action_signature = (step.action, json.dumps(step.action_input, sort_keys=True))
        repeat_streak = repeat_streak + 1 if action_signature == last_action_signature else 1
        last_action_signature = action_signature
        if repeat_streak >= 2 and step.action != "finish":
            stopped_reason = "stuck_repeated_action"
            print(f"[step {step_num}] Stopping -- same action repeated with identical input two steps in a row.")
            break

        if step.action == "finish":
            answer = step.action_input.get("answer", "")
            stopped_reason = "finished"
            transcript_lines.append(f"Thought: {step.thought}\nAction: finish\nAnswer: {answer}")
            break

        if step.action == "search_policy_docs":
            observation = tool_search_policy_docs(store, embed_model, hybrid_retriever, step.action_input)
        elif step.action == "calculate_tenure":
            observation = tool_calculate_tenure(step.action_input)
        else:
            observation = f"ERROR: unknown tool '{step.action}'"
        num_tool_calls += 1

        print(f"[step {step_num}] Observation: {observation[:200]}{'...' if len(observation) > 200 else ''}")
        transcript_lines.append(
            f"Thought: {step.thought}\nAction: {step.action}({step.action_input})\nObservation: {observation}"
        )

    if answer is None:
        answer = "Agent did not reach a final answer within the step/time limit."

    return {
        "answer": answer,
        "transcript": transcript_lines,
        "num_llm_calls": num_llm_calls,
        "num_tool_calls": num_tool_calls,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "stopped_reason": stopped_reason,
        "num_steps": len(transcript_lines),
    }


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else input("Question> ").strip()
    result = run_agent(question)

    print(f"\n{'='*80}")
    print(f"FINAL ANSWER ({result['stopped_reason']}, {result['num_steps']} steps, "
          f"{result['num_llm_calls']} LLM calls, {result['num_tool_calls']} tool calls, "
          f"{result['elapsed_seconds']}s):")
    print(result["answer"])
