import json
import time
from pathlib import Path

import pytest

from env_rl.monitor.logging import (
    ChainedJsonlWriter,
    ChainVerificationError,
    verify,
)


def test_seq_starts_at_zero_and_increments(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    for _ in range(4):
        writer.append({"x": 1})
    writer.close()

    seqs = [json.loads(line)["seq"] for line in path.read_text().splitlines()]
    assert seqs == [0, 1, 2, 3]


def test_verify_detects_gap(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    for _ in range(4):
        writer.append({"x": 1})
    writer.close()

    lines = path.read_text().splitlines()
    del lines[2]  # gap: seq jumps from 1 to 3
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ChainVerificationError):
        verify(path, root_hash=root_hash)


def test_timestamps_are_monotonic_non_decreasing(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    writer.append({"x": 1})
    time.sleep(0.001)
    writer.append({"x": 2})
    time.sleep(0.001)
    writer.append({"x": 3})
    writer.close()

    timestamps = [json.loads(line)["ts"] for line in path.read_text().splitlines()]
    assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


def test_verify_detects_non_monotonic_ts(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    writer.append({"x": 1})
    writer.append({"x": 2})
    writer.close()

    lines = path.read_text().splitlines()
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    rec0["ts"], rec1["ts"] = rec1["ts"] + 10, rec1["ts"]  # flip order
    lines[0] = json.dumps(rec0)
    lines[1] = json.dumps(rec1)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ChainVerificationError):
        verify(path, root_hash=root_hash)
