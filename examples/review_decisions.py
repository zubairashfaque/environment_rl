"""Interactive review tool: walk one attempt's decisions, mark each gold/bad/skip.

Usage:
    poetry run python examples/review_decisions.py llm_runs_real/attempt_01
    poetry run python examples/review_decisions.py llm_runs_real/attempt_01 --reviewer zubair
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env_rl.harness.human_review import HumanReview, append_review, load_reviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--reviewer", default="user")
    parser.add_argument(
        "--only-new", action="store_true",
        help="Skip decisions already reviewed in decision_review.jsonl",
    )
    args = parser.parse_args()

    if not args.attempt_dir.exists():
        print(f"error: {args.attempt_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    decisions_path = args.attempt_dir / "judge_logs" / "decision_log.jsonl"
    if not decisions_path.exists():
        print(f"error: {decisions_path} does not exist", file=sys.stderr)
        sys.exit(2)

    reviews_path = args.attempt_dir / "decision_review.jsonl"
    existing = load_reviews(reviews_path)
    seen_keys = {(r.epoch, tuple(r.decision.get("cites", []))) for r in existing}

    decisions = []
    for line in decisions_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        payload = rec.get("payload", {})
        if payload.get("kind") != "decision":
            continue
        decisions.append(payload)

    print(f"Reviewing {len(decisions)} decisions in {args.attempt_dir}")
    if args.only_new:
        print(f"(skipping {len(seen_keys)} already-reviewed)")

    for d in decisions:
        key = (int(d.get("epoch", -1)), tuple(d.get("cites", [])))
        if args.only_new and key in seen_keys:
            continue
        print("\n" + "=" * 68)
        print(f"  Epoch {d.get('epoch')}  cites={d.get('cites')}")
        print(f"  event_type: {d.get('event_type')}")
        print(f"  remedy_direction: {d.get('remedy_direction')}")
        print(f"  justification: {d.get('justification', '')[:140]}")
        if "edit" in d:
            print(f"  edit: {d['edit']}")
        choice = input("[g]old / [b]ad / [s]kip / [q]uit: ").strip().lower()
        if choice.startswith("q"):
            break
        verdict = {"g": "gold", "b": "bad", "s": "skip"}.get(choice[:1], "skip")
        notes = ""
        if verdict in ("gold", "bad"):
            notes = input("  notes (optional): ").strip()
        append_review(reviews_path, HumanReview(
            attempt_dir=str(args.attempt_dir), epoch=int(d.get("epoch", 0)),
            decision=d, verdict=verdict, notes=notes, reviewer=args.reviewer,
        ))
        print(f"  recorded: {verdict}")

    print(f"\nAll reviews written to {reviews_path}")


if __name__ == "__main__":
    main()
