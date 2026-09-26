"""Week 8 trajectory evaluation for the HR-policy agent.

Usage:
  python agent_trajectory_eval.py --fixtures
  python agent_trajectory_eval.py --live --save week8_before.json

``--fixtures`` never calls Qdrant or an LLM.  It proves evaluator behavior and
the injection boundary.  ``--live`` is the evidence-producing run: execute it
before and after one focused fix, then compare the saved JSON summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_security import sanitize_untrusted_document_text


CASES = [
    {
        "id": "basic_policy_lookup",
        "question": "How many days of paid sick leave do employees get per year?",
        "expected_keywords": ["10 days", "10 paid days"],
        "required_actions": ["search_policy_docs", "finish"],
        "forbidden_actions": ["calculate_tenure"],
    },
    {
        "id": "tenure_reasoning",
        "question": "A new employee is 45 days into their job. Can they work remotely now?",
        "expected_keywords": ["not eligible", "not allowed", "90"],
        "required_actions": ["search_policy_docs", "calculate_tenure", "finish"],
        "forbidden_actions": [],
    },
]


def _contains_in_order(actions: list[str], required: list[str]) -> bool:
    cursor = 0
    for action in actions:
        if cursor < len(required) and action == required[cursor]:
            cursor += 1
    return cursor == len(required)


def evaluate_trajectory(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Score outcome and path separately; a correct answer cannot hide a bad path."""
    events = result.get("events", [])
    actions = [event.get("action") for event in events]
    answer = result.get("answer", "")
    outcome_ok = any(keyword.lower() in answer.lower() for keyword in case["expected_keywords"])
    required_ok = _contains_in_order(actions, case["required_actions"])
    forbidden_used = sorted(set(actions) & set(case["forbidden_actions"]))
    invalid_inputs = [event["step"] for event in events if event.get("input_valid") is False]
    trajectory_ok = required_ok and not forbidden_used and not invalid_inputs and result.get("stopped_reason") == "finished"
    return {
        "id": case["id"], "outcome_ok": outcome_ok, "trajectory_ok": trajectory_ok,
        "outcome_trajectory_gap": outcome_ok and not trajectory_ok,
        "actions": actions, "forbidden_used": forbidden_used, "invalid_input_steps": invalid_inputs,
        "stopped_reason": result.get("stopped_reason"),
    }


def run_fixture_checks() -> list[dict[str, Any]]:
    """Deterministic self-checks, including the required correct-outcome/wrong-path case."""
    wrong_path = {
        "answer": "The employee is not eligible until the 90-day probation period ends.",
        "stopped_reason": "finished",
        "events": [
            {"step": 1, "action": "search_policy_docs", "input_valid": True},
            {"step": 2, "action": "finish", "input_valid": True},
        ],
    }
    trajectory = evaluate_trajectory(wrong_path, CASES[1])
    if not trajectory["outcome_trajectory_gap"]:
        raise AssertionError("Fixture must prove a correct outcome can have a wrong trajectory.")

    fixture_path = Path("fixtures/week8_indirect_prompt_injection.md")
    safe_text, removed = sanitize_untrusted_document_text(fixture_path.read_text(encoding="utf-8"))
    if removed < 1 or "ignore previous instructions" in safe_text.lower():
        raise AssertionError("Injection fixture was not quarantined by the tool-output boundary.")

    return [
        {"check": "outcome_vs_trajectory_gap", "passed": True, "detail": trajectory},
        {"check": "indirect_prompt_injection_boundary", "passed": True, "removed_lines": removed},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", action="store_true", help="run deterministic evaluator/security checks")
    parser.add_argument("--live", action="store_true", help="run the real agent against CASES")
    parser.add_argument("--save", metavar="PATH", help="save live summary JSON for before/after comparison")
    args = parser.parse_args()
    if not args.fixtures and not args.live:
        parser.error("choose --fixtures, --live, or both")

    report: dict[str, Any] = {}
    if args.fixtures:
        report["fixtures"] = run_fixture_checks()
        print(json.dumps(report["fixtures"], indent=2))

    if args.live:
        from agent import run_agent
        results = []
        for case in CASES:
            scored = evaluate_trajectory(run_agent(case["question"]), case)
            results.append(scored)
            print(f"{case['id']}: outcome={'PASS' if scored['outcome_ok'] else 'FAIL'} "
                  f"trajectory={'PASS' if scored['trajectory_ok'] else 'FAIL'}")
        report["live"] = results
        report["summary"] = {
            "cases": len(results),
            "outcome_pass_rate": sum(r["outcome_ok"] for r in results) / len(results),
            "trajectory_pass_rate": sum(r["trajectory_ok"] for r in results) / len(results),
            "outcome_trajectory_gaps": sum(r["outcome_trajectory_gap"] for r in results),
        }
        print(json.dumps(report["summary"], indent=2))

    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
