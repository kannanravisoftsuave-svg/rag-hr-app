"""
Compare two saved eval.py runs, per problem type, to prove a change helped.

Usage:
  python eval.py --save before.json          # run against the current code
  ... make your one improvement ...
  python eval.py --save after.json           # run again
  python compare_eval_runs.py before.json after.json
"""
import argparse
import json


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)

    print(f"\n{'Tag (problem type)':<50} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 82)

    all_tags = sorted(set(before["tag_stats"]) | set(after["tag_stats"]))
    for tag in all_tags:
        b = before["tag_stats"].get(tag)
        a = after["tag_stats"].get(tag)
        b_str = f"{b['passed']}/{b['total']}" if b else "n/a"
        a_str = f"{a['passed']}/{a['total']}" if a else "n/a"
        if b and a and b["total"] and a["total"]:
            b_rate, a_rate = b["passed"] / b["total"], a["passed"] / a["total"]
            delta = a_rate - b_rate
            change = f"{delta:+.0%}" if delta != 0 else "same"
        else:
            change = "n/a"
        print(f"{tag:<50} {b_str:>10} {a_str:>10} {change:>10}")

    print("-" * 82)
    b_overall = before["passed"] / before["total"] if before["total"] else 0
    a_overall = after["passed"] / after["total"] if after["total"] else 0
    print(f"{'OVERALL answer score':<50} {before['passed']}/{before['total']:>7} "
          f"{after['passed']}/{after['total']:>8}   {a_overall - b_overall:+.0%}")
    print(f"{'hit-rate@3':<50} {before['hit_rate_3']:>10.0%} {after['hit_rate_3']:>10.0%}   "
          f"{after['hit_rate_3'] - before['hit_rate_3']:+.0%}")


if __name__ == "__main__":
    main()
