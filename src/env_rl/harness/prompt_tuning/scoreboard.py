"""Persistent scoreboard of which prompt-tuning techniques win over time.

Lets the Tuner bias technique selection based on past success. Stored as a
simple JSON file at ``~/.env_rl/scoreboard.json`` (or a user-supplied path);
survives across runs so the system gets smarter with use.
"""

from __future__ import annotations

import json
from pathlib import Path


class TechniqueScoreboard:
    """Tracks technique wins/losses/ties across meta-loop iterations."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path.home() / ".env_rl" / "scoreboard.json"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, int]] = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def record(self, technique: str, outcome: str) -> None:
        """Record one iteration. ``outcome`` is 'win', 'loss', or 'tie'."""
        if outcome not in {"win", "loss", "tie"}:
            raise ValueError(f"outcome must be win|loss|tie, got {outcome!r}")
        entry = self._data.setdefault(technique, {"win": 0, "loss": 0, "tie": 0})
        entry[outcome] += 1
        self._save()

    def win_rate(self, technique: str) -> float:
        """Wins / (wins + losses). Returns 0.5 if no data."""
        entry = self._data.get(technique)
        if not entry:
            return 0.5
        denom = entry.get("win", 0) + entry.get("loss", 0)
        if denom == 0:
            return 0.5
        return entry.get("win", 0) / denom

    def summary(self) -> dict[str, dict[str, int | float]]:
        """Readable snapshot."""
        out: dict[str, dict[str, int | float]] = {}
        for technique, entry in self._data.items():
            total = sum(entry.values())
            out[technique] = {
                **entry,
                "total": total,
                "win_rate": self.win_rate(technique),
            }
        return out
