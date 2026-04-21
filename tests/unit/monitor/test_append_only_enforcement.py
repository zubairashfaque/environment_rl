import io
import json
from pathlib import Path

import pytest

from env_rl.monitor.logging import ChainedJsonlWriter


def test_writer_uses_append_only_mode(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    try:
        assert writer._fh.mode == "ab"
    finally:
        writer.close()


def test_append_only_mode_prevents_overwrite(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    writer.append({"x": 1})
    original_first_byte = path.read_bytes()[:1]
    # Adversarial seek+write: POSIX guarantees writes go to EOF in "ab" mode.
    try:
        writer._fh.seek(0)
        writer._fh.write(b"X")
        writer._fh.flush()
    except (OSError, io.UnsupportedOperation):
        pass
    writer.close()
    assert path.read_bytes()[:1] == original_first_byte


def test_writer_cannot_truncate_existing_entries(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    writer = ChainedJsonlWriter(path, root_hash=root_hash)
    writer.append({"x": 1})
    writer.append({"x": 2})
    writer.close()

    before = path.read_text()
    # Re-opening the writer must NOT clear prior entries;
    # it must resume with the correct next seq and prev_hash.
    writer2 = ChainedJsonlWriter(path, root_hash=root_hash)
    writer2.append({"x": 3})
    writer2.close()

    lines = path.read_text().splitlines()
    assert before.splitlines() == lines[:2]
    assert json.loads(lines[2])["seq"] == 2
    assert json.loads(lines[2])["prev_hash"] == json.loads(lines[1])["hash"]


def test_writer_raises_if_path_parent_missing(tmp_log_dir: Path, root_hash: str) -> None:
    path = tmp_log_dir / "nope" / "metrics.jsonl"
    with pytest.raises(FileNotFoundError):
        ChainedJsonlWriter(path, root_hash=root_hash)


def test_writer_context_manager_closes_handle(
    tmp_log_dir: Path, root_hash: str
) -> None:
    path = tmp_log_dir / "metrics.jsonl"
    with ChainedJsonlWriter(path, root_hash=root_hash) as writer:
        writer.append({"x": 1})
        fh = writer._fh
    assert fh.closed
