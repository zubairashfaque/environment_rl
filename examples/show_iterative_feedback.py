"""Show the iterative self-refine feedback fed into each attempt.

The system prompt of every attempt after the first includes a summary of
prior attempts' scores and violations — that's how the model "learns"
between attempts without any weight update. This script extracts and
pretty-prints that feedback block per attempt so the loop is legible.

Usage::

    poetry run python examples/show_iterative_feedback.py llm_runs
    poetry run python examples/show_iterative_feedback.py llm_runs_real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"error: {args.base_dir} not found", file=sys.stderr)
        sys.exit(2)

    attempts = sorted(args.base_dir.glob("attempt_*"))
    if not attempts:
        print(f"no attempt_NN/ directories in {args.base_dir}", file=sys.stderr)
        sys.exit(2)

    for ad in attempts:
        idx = ad.name.split("_")[-1]
        feedback_path = ad / "feedback_in.json"
        summary_path = ad / "summary.json"

        if not feedback_path.exists():
            print(f"attempt {idx}: (no feedback_in.json — run with a newer harness)")
            continue

        feedback = json.loads(feedback_path.read_text())
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

        print("=" * 70)
        print(f"ATTEMPT {idx}")
        print("=" * 70)

        if summary:
            s = summary["scores"]
            print(
                f"Scores: accuracy={s['accuracy_score']:.3f} "
                f"process={s['process_score']:.3f} "
                f"test_acc={s['test_accuracy']:.3f} "
                f"violations={s['violations']}/{s['total_decisions']} "
                f"hard_fail={s['hard_fail']}"
            )

        priors = feedback.get("prior_attempts", [])
        if not priors:
            print("Feedback fed IN to this attempt: (none — first attempt)")
        else:
            print(f"Feedback fed IN to this attempt ({len(priors)} prior attempt(s)):")
            for p in priors:
                print(
                    f"  • attempt {p['attempt_index']}: "
                    f"accuracy={p['accuracy_score']:.3f} "
                    f"process={p['process_score']:.3f} "
                    f"violations={p['violations']}/{p['total_decisions']}"
                )
                for v in p.get("violation_summary", [])[:5]:
                    print(
                        f"      - {v['kind']} rule={v['rule']} epoch={v['epoch']}"
                    )
                if len(p.get("violation_summary", [])) > 5:
                    rest = len(p["violation_summary"]) - 5
                    print(f"      ... + {rest} more")
        print()


if __name__ == "__main__":
    main()
