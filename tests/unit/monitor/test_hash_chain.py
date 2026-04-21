import hashlib
import json
from pathlib import Path

import pytest

from env_rl.monitor.logging import (
    ChainedJsonlWriter,
    ChainVerificationError,
    verify,
)


def test_two_sequential_writes_are_linked(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    try:
        writer.append({"kind": "epoch", "epoch": 0, "loss": 2.3})
        writer.append({"kind": "epoch", "epoch": 1, "loss": 2.1})
    finally:
        writer.close()

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["prev_hash"] == root_hash
    assert lines[1]["prev_hash"] == lines[0]["hash"]
    assert lines[0]["seq"] == 0
    assert lines[1]["seq"] == 1


def test_hash_deterministic_over_payload_seq_and_ts(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    writer.append({"x": 1})
    writer.close()

    rec = json.loads(path.read_text().splitlines()[0])
    payload_canonical = json.dumps(rec["payload"], sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(
        (root_hash + payload_canonical + str(rec["seq"]) + str(rec["ts"])).encode("utf-8")
    ).hexdigest()
    assert rec["hash"] == expected


def test_verify_passes_on_clean_log(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    for i in range(5):
        writer.append({"epoch": i})
    writer.close()

    verify(path, root_hash=root_hash)  # should not raise


def test_verify_detects_tampered_payload(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    for i in range(3):
        writer.append({"epoch": i})
    writer.close()

    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["epoch"] = 99  # tamper
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ChainVerificationError):
        verify(path, root_hash=root_hash)


def test_verify_detects_tampered_hash(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    for i in range(3):
        writer.append({"epoch": i})
    writer.close()

    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["hash"] = "f" * 64  # tamper downstream chain by replacing hash
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ChainVerificationError):
        verify(path, root_hash=root_hash)


def test_verify_detects_wrong_root_hash(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    writer.append({"epoch": 0})
    writer.close()

    with pytest.raises(ChainVerificationError):
        verify(path, root_hash="1" * 64)
