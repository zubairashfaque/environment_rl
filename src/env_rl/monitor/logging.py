from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, BinaryIO


class ChainVerificationError(Exception):
    """Raised when a hash-chained log fails integrity verification."""


_HASH_LEN = 64  # sha256 hex digest length


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compute_hash(prev_hash: str, payload: dict[str, Any], seq: int, ts: float) -> str:
    material = prev_hash + _canonical(payload) + str(seq) + str(ts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ChainedJsonlWriter:
    """Append-only JSONL writer with a SHA-256 hash chain.

    Each line is a JSON object with fields:
      - seq: monotonic integer starting at 0
      - ts: float wall-clock seconds (monotonic across calls)
      - prev_hash: hash of the previous line (or root_hash for seq=0)
      - payload: caller-supplied dict
      - hash: sha256(prev_hash || canonical(payload) || str(seq) || str(ts))

    The file is opened in "ab" (binary append) mode — POSIX guarantees every
    write lands at EOF regardless of seek(), so past lines cannot be
    overwritten through this handle.
    """

    def __init__(self, path: Path, *, root_hash: str) -> None:
        if len(root_hash) != _HASH_LEN:
            raise ValueError(f"root_hash must be {_HASH_LEN} hex chars, got {len(root_hash)}")
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"parent directory missing: {path.parent}")
        self._path = path
        self._root_hash = root_hash
        self._seq, self._prev_hash, self._last_ts = self._resume_state()
        self._fh: BinaryIO = open(path, "ab")  # noqa: SIM115

    def _resume_state(self) -> tuple[int, str, float]:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return 0, self._root_hash, 0.0
        last_rec: dict[str, Any] | None = None
        with open(self._path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_rec = json.loads(line)
        assert last_rec is not None
        return int(last_rec["seq"]) + 1, str(last_rec["hash"]), float(last_rec["ts"])

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        ts = max(time.time(), self._last_ts)
        seq = self._seq
        h = _compute_hash(self._prev_hash, payload, seq, ts)
        record = {
            "seq": seq,
            "ts": ts,
            "prev_hash": self._prev_hash,
            "payload": payload,
            "hash": h,
        }
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        self._fh.write(line)
        self._fh.flush()
        self._seq += 1
        self._prev_hash = h
        self._last_ts = ts
        return record

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> ChainedJsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verify(path: Path, *, root_hash: str) -> None:
    """Walk the chain end-to-end; raise ChainVerificationError on any issue."""
    path = Path(path)
    if not path.exists():
        raise ChainVerificationError(f"log file does not exist: {path}")
    if len(root_hash) != _HASH_LEN:
        raise ChainVerificationError(f"root_hash must be {_HASH_LEN} hex chars")

    prev_hash = root_hash
    expected_seq = 0
    last_ts = 0.0

    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ChainVerificationError(f"line {lineno}: invalid JSON: {e}") from e

            for field in ("seq", "ts", "prev_hash", "payload", "hash"):
                if field not in rec:
                    raise ChainVerificationError(f"line {lineno}: missing field {field!r}")

            if rec["seq"] != expected_seq:
                raise ChainVerificationError(
                    f"line {lineno}: seq gap (expected {expected_seq}, got {rec['seq']})"
                )
            if rec["prev_hash"] != prev_hash:
                raise ChainVerificationError(
                    f"line {lineno}: prev_hash mismatch at seq={rec['seq']}"
                )
            if float(rec["ts"]) < last_ts:
                raise ChainVerificationError(
                    f"line {lineno}: timestamp went backwards ({rec['ts']} < {last_ts})"
                )

            expected = _compute_hash(
                rec["prev_hash"], rec["payload"], rec["seq"], rec["ts"]
            )
            if expected != rec["hash"]:
                raise ChainVerificationError(
                    f"line {lineno}: hash mismatch at seq={rec['seq']} "
                    f"(expected {expected}, stored {rec['hash']})"
                )

            prev_hash = rec["hash"]
            last_ts = float(rec["ts"])
            expected_seq += 1
