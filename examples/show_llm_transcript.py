"""Pretty-print the LLM conversation log for one attempt.

Usage::

    poetry run python examples/show_llm_transcript.py llm_runs/attempt_02/llm_transcript.jsonl
    poetry run python examples/show_llm_transcript.py llm_runs/attempt_02/llm_transcript.jsonl --no-system
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path, help="path to llm_transcript.jsonl")
    parser.add_argument(
        "--no-system", action="store_true",
        help="skip the system prompt (it's the same for every call in one attempt)",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="only show the call at this epoch",
    )
    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"error: {args.transcript} not found", file=sys.stderr)
        sys.exit(2)

    for raw in args.transcript.read_text().splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        kind = rec.get("kind")

        if kind == "system_prompt" and not args.no_system:
            print("=" * 70)
            print(f"SYSTEM PROMPT  (model={rec.get('model')}, T={rec.get('temperature')})")
            print("=" * 70)
            print(rec["text"])
            print()

        elif kind == "call":
            if args.epoch is not None and rec.get("epoch") != args.epoch:
                continue
            fired = [r for r, v in rec.get("all_fired", {}).items() if v]
            print("-" * 70)
            print(
                f"EPOCH {rec.get('epoch')}  "
                f"top_rule={rec.get('top_rule')}  "
                f"fired={fired}"
            )
            print("-" * 70)
            print(">>> USER MESSAGE")
            print(rec.get("user_message", "").rstrip())
            print()
            print("<<< LLM RESPONSE")
            try:
                parsed = json.loads(rec.get("response", ""))
                print(json.dumps(parsed, indent=2))
            except (json.JSONDecodeError, TypeError):
                print(rec.get("response", ""))
            print()


if __name__ == "__main__":
    main()
