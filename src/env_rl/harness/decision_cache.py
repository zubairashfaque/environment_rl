"""In-memory cache for decisions made on identical diagnostic states.

During training, the same ``(top_rule, all_fired_fingerprint)`` combination
frequently repeats across consecutive epochs (EMA keeps a rule firing after
a remedy has already been applied). Caching the decision avoids burning a
fresh OpenAI call on what is almost certainly the same answer.

This is a best-effort cache: if the cached decision turns out to be wrong
for the current state, the judge still catches it as a defensibility
violation. The cache only saves tokens, not judgment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def fingerprint(
    *,
    top_rule: str,
    all_fired: dict[str, bool],
    current_lr: float,
    current_batch_size: int,
) -> str:
    """Stable hash of the decision-relevant inputs (excluding metrics values).

    Metrics are noisy floats; including them would never match. The rule
    firings + hparams are the decision-relevant invariants.
    """
    fired_tuple = tuple(sorted(r for r, v in all_fired.items() if v))
    material = f"{top_rule}|{fired_tuple}|{round(current_lr, 6)}|{current_batch_size}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass
class CachedDecision:
    fingerprint: str
    decision: Any  # the Decision object
    hits: int = 0


class DecisionCache:
    """TTL-less, per-session cache of decisions keyed by fingerprint."""

    def __init__(self, *, max_size: int = 64) -> None:
        self._cache: dict[str, CachedDecision] = {}
        self._max_size = max_size
        self._total_hits = 0
        self._total_misses = 0

    def get(self, fp: str) -> Any | None:
        entry = self._cache.get(fp)
        if entry is None:
            self._total_misses += 1
            return None
        entry.hits += 1
        self._total_hits += 1
        return entry.decision

    def put(self, fp: str, decision: Any) -> None:
        if len(self._cache) >= self._max_size:
            # Evict the least-recently-added (dict preserves insertion order
            # in Python 3.7+). Not true LRU but simpler and good enough.
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[fp] = CachedDecision(fingerprint=fp, decision=decision)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "hits": self._total_hits,
            "misses": self._total_misses,
            "hit_rate": (
                self._total_hits / (self._total_hits + self._total_misses)
                if (self._total_hits + self._total_misses) > 0 else 0.0
            ),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._total_hits = 0
        self._total_misses = 0
