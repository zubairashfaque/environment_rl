"""Reset prompt-tuning state in a base_dir, leaving attempt_NN/ artifacts intact.

Usage:
    poetry run python examples/reset_prompt_tuning.py llm_runs_real
    poetry run python examples/reset_prompt_tuning.py llm_runs_real --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PATHS_TO_WIPE = ("prompts", ".scoreboard.json", "meta_loop_log.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"error: {args.base_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    to_delete: list[Path] = []
    for name in PATHS_TO_WIPE:
        p = args.base_dir / name
        if p.exists():
            to_delete.append(p)

    if not to_delete:
        print(f"Nothing to reset at {args.base_dir} — no prompt-tuning state found.")
        return

    print(f"About to delete the following in {args.base_dir}:")
    for p in to_delete:
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"  {p}/ (directory, {size} bytes)")
        else:
            print(f"  {p} ({p.stat().st_size} bytes)")

    if not args.yes:
        ans = input("Proceed? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return

    for p in to_delete:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  deleted {p}")

    print("Done. attempt_NN/ folders were left intact.")


if __name__ == "__main__":
    main()
