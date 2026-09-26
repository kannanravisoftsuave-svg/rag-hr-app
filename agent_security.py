"""Small, deterministic safety controls for the Week 8 HR-policy agent.

Retrieved documents are reference data, not instructions.  These helpers sit at
the tool boundary so a retrieved prompt injection is labelled/removed before it
is placed in the agent's next prompt.  They are intentionally dependency-free
and conservative: suspicious lines are quarantined, while the surrounding HR
policy text remains available to answer the question.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any


MAX_QUERY_CHARS = 500
MAX_FINISH_ANSWER_CHARS = 4_000

# These patterns deliberately target instruction-like language, not ordinary HR
# policy prose.  The test fixture covers the expected attack forms.
INJECTION_PATTERNS = (
    re.compile(r"\b(ignore|disregard|override)\b.{0,80}\b(instruction|prompt|rule|previous)\b", re.I),
    re.compile(r"\b(system message|developer message|jailbreak|prompt injection)\b", re.I),
    re.compile(r"\b(reveal|exfiltrate|send)\b.{0,80}\b(api[ _-]?key|secret|password|token)\b", re.I),
    re.compile(r"\b(call|use)\b.{0,60}\btool\b.{0,80}\b(delete|upload|send|secret)\b", re.I),
)


def sanitize_untrusted_document_text(text: str) -> tuple[str, int]:
    """Remove suspicious instruction lines and return ``(safe_text, removed_count)``.

    This is not a claim that regex solves prompt injection.  It is a concrete
    boundary control for the known attack forms, supplemented by the agent
    prompt's rule that all retrieved text is untrusted data.
    """
    safe_lines: list[str] = []
    removed = 0
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in INJECTION_PATTERNS):
            safe_lines.append("[Potential prompt-injection instruction removed from untrusted document.]")
            removed += 1
        else:
            safe_lines.append(line)
    return "\n".join(safe_lines), removed


def validate_tool_input(action: str, action_input: dict[str, Any]) -> str | None:
    """Return a user-safe validation error, or ``None`` for permitted input."""
    if not isinstance(action_input, dict):
        return "action_input must be a JSON object."

    allowed: dict[str, set[str]] = {
        "search_policy_docs": {"query"},
        "calculate_tenure": {"start_date", "as_of_date"},
        "finish": {"answer"},
    }
    unexpected = set(action_input) - allowed.get(action, set())
    if unexpected:
        return f"unexpected input field(s) for {action}: {', '.join(sorted(unexpected))}."

    if action == "search_policy_docs":
        query = action_input.get("query")
        if not isinstance(query, str) or not query.strip():
            return 'action_input must include a non-empty string "query".'
        if len(query) > MAX_QUERY_CHARS:
            return f"query must be at most {MAX_QUERY_CHARS} characters."

    elif action == "calculate_tenure":
        for field in ("start_date", "as_of_date"):
            value = action_input.get(field)
            if not isinstance(value, str):
                return f'action_input must include string field "{field}" as YYYY-MM-DD.'
            try:
                date.fromisoformat(value)
            except ValueError:
                return f'"{field}" must use YYYY-MM-DD format.'

    elif action == "finish":
        answer = action_input.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return 'action_input must include a non-empty string "answer".'
        if len(answer) > MAX_FINISH_ANSWER_CHARS:
            return f"answer must be at most {MAX_FINISH_ANSWER_CHARS} characters."

    return None
