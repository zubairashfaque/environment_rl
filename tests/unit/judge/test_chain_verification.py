import json
from pathlib import Path

import pytest

from env_rl.judge.chain import verify_all_logs, verify_bookends, verify_log_chain
from env_rl.judge.deliverables import HardFail
from env_rl.monitor.logging import ChainedJsonlWriter


def _write_clean_log(path: Path, root_hash: str, n_middle: int = 3) -> None:
    w = ChainedJsonlWriter(path, root_hash=root_hash)
    w.append({"kind": "session_start", "run_config": {"seed": 42}})
    for i in range(n_middle):
        w.append({"kind": "epoch", "epoch": i})
    w.append({"kind": "session_end"})
    w.close()


def _write_three_clean_logs(dir_: Path, root_hash: str) -> None:
    for name in ("metrics_log.jsonl", "decision_log.jsonl", "rule_evaluations.jsonl"):
        _write_clean_log(dir_ / name, root_hash)


def test_verify_log_chain_passes_on_clean_log(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics_log.jsonl"
    _write_clean_log(path, root_hash)
    verify_log_chain(path, root_hash=root_hash)  # does not raise


def test_verify_log_chain_hard_fails_on_tampered_hash(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics_log.jsonl"
    _write_clean_log(path, root_hash)
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["hash"] = "f" * 64
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(HardFail):
        verify_log_chain(path, root_hash=root_hash)


def test_verify_log_chain_hard_fails_on_gap(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics_log.jsonl"
    _write_clean_log(path, root_hash)
    lines = path.read_text().splitlines()
    del lines[2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(HardFail):
        verify_log_chain(path, root_hash=root_hash)


def test_verify_log_chain_hard_fails_on_non_monotonic_ts(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics_log.jsonl"
    _write_clean_log(path, root_hash)
    lines = path.read_text().splitlines()
    rec0 = json.loads(lines[1])
    rec1 = json.loads(lines[2])
    rec0["ts"] = rec1["ts"] + 10
    lines[1] = json.dumps(rec0)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(HardFail):
        verify_log_chain(path, root_hash=root_hash)


def test_bookends_hard_fail_on_missing_session_end(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics_log.jsonl"
    w = ChainedJsonlWriter(path, root_hash=root_hash)
    w.append({"kind": "session_start", "run_config": {}})
    w.append({"kind": "epoch", "epoch": 0})
    w.close()  # no session_end
    with pytest.raises(HardFail):
        verify_bookends(path)


def test_bookends_hard_fail_on_missing_session_start(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics_log.jsonl"
    w = ChainedJsonlWriter(path, root_hash=root_hash)
    w.append({"kind": "epoch", "epoch": 0})
    w.append({"kind": "session_end"})
    w.close()
    with pytest.raises(HardFail):
        verify_bookends(path)


def test_verify_all_logs_returns_session_start(
    tmp_log_dir: Path, root_hash: str
) -> None:
    _write_three_clean_logs(tmp_log_dir, root_hash)
    start = verify_all_logs(tmp_log_dir, root_hash=root_hash)
    assert start["kind"] == "session_start"
    assert start["run_config"] == {"seed": 42}


def test_verify_all_logs_hard_fails_if_missing(
    tmp_log_dir: Path, root_hash: str
) -> None:
    _write_clean_log(tmp_log_dir / "metrics_log.jsonl", root_hash)
    # decision_log.jsonl missing
    with pytest.raises(HardFail):
        verify_all_logs(tmp_log_dir, root_hash=root_hash)
