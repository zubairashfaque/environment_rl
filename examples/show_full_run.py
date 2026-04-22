"""One-shot diagnostic viewer: print every log for one attempt.

Combines training trace, judge trace, LLM transcript, decision log, scores,
and iterative feedback into a single readable summary.

Usage::

    poetry run python examples/show_full_run.py llm_runs/attempt_01
    poetry run python examples/show_full_run.py llm_runs/attempt_02 --section judge
    poetry run python examples/show_full_run.py llm_runs/attempt_02 --section llm --epoch 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def _subsection(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 66 - len(title)))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _show_summary(attempt_dir: Path) -> None:
    s = attempt_dir / "summary.json"
    if not s.exists():
        print(f"(no summary.json in {attempt_dir})")
        return
    data = json.loads(s.read_text())
    _section("SUMMARY")
    sc = data["scores"]
    print(f"  mode: {data.get('mode', '?')}")
    print(f"  attempt_index: {data.get('attempt_index', '?')}")
    print(f"  hard_fail: {sc['hard_fail']}")
    print(f"  accuracy_score: {sc['accuracy_score']:.4f}")
    print(f"  process_score:  {sc['process_score']:.4f}")
    print(f"  test_accuracy:  {sc['test_accuracy']:.4f}")
    print(f"  violations:     {sc['violations']} / {sc['total_decisions']}")
    vs = data.get("violation_summary", [])
    if vs:
        print(f"  {len(vs)} violations:")
        from collections import Counter
        c = Counter((v["kind"], v["rule"]) for v in vs)
        for (kind, rule), n in c.most_common():
            print(f"    {n:>3d}  {kind:25s} rule={rule}")


def _show_judge(attempt_dir: Path) -> None:
    p = attempt_dir / "judge_trace.json"
    _section("JUDGE TRACE (11-step audit)")
    if not p.exists():
        print(f"(no judge_trace.json — run with the updated harness)")
        return
    data = json.loads(p.read_text())
    for step in data["steps"]:
        status = step["status"]
        marker = {"pass": "✓", "hard_fail": "✗", "skipped": "-"}.get(status, "?")
        color = {"pass": "", "hard_fail": "", "skipped": ""}[status]
        print(f"  [{marker}] step {step['step']:>2d}  {step['name']:30s} "
              f"{step['kind']:18s}  {step['duration_ms']:7.2f} ms")
        if step.get("detail"):
            print(f"        detail: {step['detail']}")
        if step.get("extra"):
            for k, v in step["extra"].items():
                print(f"        {k}: {v}")


def _show_training(attempt_dir: Path) -> None:
    p = attempt_dir / "training_trace.jsonl"
    _section("TRAINING TRACE (per-epoch)")
    if not p.exists():
        print(f"(no training_trace.jsonl — run with the updated harness)")
        return
    for rec in _read_jsonl(p):
        kind = rec.get("kind")
        if kind == "session_start":
            cfg = rec["cfg"]
            print(f"  START  seed={cfg['seed']} lr={cfg['lr']} bs={cfg['batch_size']} "
                  f"max_epochs={cfg['max_epochs']} num_blocks={cfg['num_blocks']} "
                  f"activation={cfg['activation']}")
        elif kind == "epoch":
            fired = rec.get("fired_rules", [])
            fired_str = "{" + ",".join(fired) + "}" if fired else "{}"
            print(f"  ep{rec['epoch']:>3d}  loss={rec.get('train_loss', 0):.3f}/"
                  f"{rec.get('val_loss', 0):.3f}  acc={rec.get('train_acc', 0):.3f}/"
                  f"{rec.get('val_acc', 0):.3f}  lr={rec.get('lr', 0):.4g}  "
                  f"grad_max={rec.get('max_grad', 0):.3f}  "
                  f"dead_relu={rec.get('dead_relu', 0):.2f}  fired={fired_str}")
        elif kind == "decision":
            print(f"       DECISION  cite={rec['cited_rule']}  event={rec['event_type']}  "
                  f"direction={rec.get('remedy_direction')}  "
                  f"just=\"{rec['justification'][:80]}\"")
        elif kind == "remedy_applied":
            print(f"       REMEDY    {rec['change']} -> {rec['to']}")
        elif kind == "session_end":
            print(f"  END    best_val_acc={rec.get('best_val_acc', 0):.4f}")


def _show_llm(attempt_dir: Path, filter_epoch: int | None) -> None:
    p = attempt_dir / "llm_transcript.jsonl"
    _section("LLM TRANSCRIPT")
    if not p.exists():
        print(f"(no llm_transcript.jsonl — either not run or scripted policy used)")
        return
    records = _read_jsonl(p)
    total_tokens = 0
    for rec in records:
        if rec.get("kind") == "system_prompt":
            print(f"  [system] model={rec['model']}  T={rec.get('temperature')}")
            print(f"  system_prompt length: {len(rec['text'])} chars")
        elif rec.get("kind") == "call":
            if filter_epoch is not None and rec.get("epoch") != filter_epoch:
                continue
            usage = rec.get("usage", {})
            tot = usage.get("total_tokens", 0)
            total_tokens += tot
            fired = [r for r, v in rec.get("all_fired", {}).items() if v]
            print(f"\n  [call] epoch={rec.get('epoch')} top_rule={rec.get('top_rule')} "
                  f"fired={fired} tokens={tot}")
            print(f"  --- user message ---")
            print("  " + rec.get("user_message", "").replace("\n", "\n  "))
            print(f"  --- response ---")
            try:
                parsed = json.loads(rec.get("response", ""))
                pretty = json.dumps(parsed, indent=2)
                print("  " + pretty.replace("\n", "\n  "))
            except (json.JSONDecodeError, TypeError):
                print("  " + rec.get("response", ""))
    if filter_epoch is None:
        call_count = sum(1 for r in records if r.get("kind") == "call")
        print(f"\n  [total]  {call_count} LLM calls, {total_tokens} tokens")


def _show_feedback(attempt_dir: Path) -> None:
    p = attempt_dir / "feedback_in.json"
    _section("ITERATIVE FEEDBACK (fed in to this attempt)")
    if not p.exists():
        print(f"(no feedback_in.json)")
        return
    data = json.loads(p.read_text())
    priors = data.get("prior_attempts", [])
    if not priors:
        print("  (no priors — this was the first attempt)")
        return
    print(f"  {len(priors)} prior attempts fed into this attempt's system prompt:")
    for p_ in priors:
        print(f"    • attempt {p_['attempt_index']}: "
              f"accuracy={p_['accuracy_score']:.3f}  process={p_['process_score']:.3f}  "
              f"violations={p_['violations']}/{p_['total_decisions']}")
        for v in p_.get("violation_summary", [])[:3]:
            print(f"      - {v['kind']} rule={v['rule']} epoch={v['epoch']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_dir", type=Path,
                        help="e.g. llm_runs/attempt_01")
    parser.add_argument("--section",
                        choices=["all", "summary", "judge", "training", "llm", "feedback"],
                        default="all")
    parser.add_argument("--epoch", type=int, default=None,
                        help="only show LLM calls at this epoch")
    args = parser.parse_args()

    if not args.attempt_dir.exists():
        print(f"error: {args.attempt_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    sections = [args.section] if args.section != "all" else [
        "summary", "feedback", "training", "judge", "llm"
    ]
    for s in sections:
        {"summary": _show_summary, "judge": _show_judge,
         "training": _show_training, "feedback": _show_feedback,
         "llm": lambda d: _show_llm(d, args.epoch)}[s](args.attempt_dir)
    print()


if __name__ == "__main__":
    main()
