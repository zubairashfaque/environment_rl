"""Judge step 5: log-chain integrity.

Each of the three judge-owned logs must have:
  - unbroken sha-256 chain rooted at the configured root_hash
  - contiguous seq numbers from 0
  - monotonically non-decreasing timestamps
  - a single matched session_start / session_end bookend pair

Any violation is a hard fail — zeroes both scores.
"""

from __future__ import annotations

import json
from pathlib import Path

from env_rl.judge.deliverables import HardFail
from env_rl.monitor.logging import ChainVerificationError, verify as _verify_chain

LOG_NAMES = ("metrics_log.jsonl", "decision_log.jsonl", "rule_evaluations.jsonl")


def verify_log_chain(path: Path, *, root_hash: str) -> None:
    """Wraps the monitor-side verify() as a HardFail-raising judge check."""
    try:
        _verify_chain(path, root_hash=root_hash)
    except ChainVerificationError as e:
        raise HardFail(f"{path.name}: {e}") from e


def verify_bookends(path: Path) -> dict:
    """Assert session_start at seq=0, session_end as the final record.

    Returns the ``session_start`` payload so downstream checks can compare
    ``run_config.json`` to the logged value.
    """
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise HardFail(f"{path.name}: empty log")
    first = json.loads(lines[0])
    last = json.loads(lines[-1])

    if first["seq"] != 0:
        raise HardFail(f"{path.name}: first record seq={first['seq']}, expected 0")
    if first["payload"].get("kind") != "session_start":
        raise HardFail(f"{path.name}: first record is not session_start")
    if last["payload"].get("kind") != "session_end":
        raise HardFail(f"{path.name}: final record is not session_end")

    # Exactly one of each
    kinds = [json.loads(ln)["payload"].get("kind") for ln in lines]
    if kinds.count("session_start") != 1:
        raise HardFail(f"{path.name}: found {kinds.count('session_start')} session_start")
    if kinds.count("session_end") != 1:
        raise HardFail(f"{path.name}: found {kinds.count('session_end')} session_end")

    return first["payload"]


def verify_all_logs(judge_logs: str | Path, *, root_hash: str) -> dict:
    """Verify chain + bookends for all three judge-owned logs.

    Returns the session_start payload from ``metrics_log.jsonl`` (all three
    share the same run_config so any is authoritative).
    """
    judge_logs = Path(judge_logs)
    session_start: dict | None = None
    for name in LOG_NAMES:
        path = judge_logs / name
        if not path.exists():
            raise HardFail(f"missing log: {path}")
        verify_log_chain(path, root_hash=root_hash)
        start = verify_bookends(path)
        if name == "metrics_log.jsonl":
            session_start = start
    assert session_start is not None
    return session_start


def read_epoch_records(path: Path) -> list[dict]:
    """Parse only the 'epoch' records from metrics_log.jsonl."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["payload"].get("kind") == "epoch":
            out.append(rec["payload"])
    return out


def read_rule_eval_records(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["payload"].get("kind") == "rule_eval":
            out.append(rec["payload"])
    return out


def read_decision_records(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["payload"].get("kind") == "decision":
            out.append(rec["payload"])
    return out
