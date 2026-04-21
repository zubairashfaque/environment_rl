from __future__ import annotations

from typing import Sequence


class EMA:
    """Exponential moving average.

    value_t = alpha * x_t + (1 - alpha) * value_{t-1}
    First update seeds `value` with the input itself.
    """

    def __init__(self, alpha: float) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._value: float | None = None

    @property
    def value(self) -> float:
        if self._value is None:
            raise RuntimeError("EMA has not been updated yet")
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = float(x)
        else:
            self._value = self._alpha * float(x) + (1.0 - self._alpha) * self._value
        return self._value


def smooth(values: Sequence[float], alpha: float) -> list[float]:
    """Produce the full EMA trace of `values` (same length)."""
    ema = EMA(alpha=alpha)
    return [ema.update(v) for v in values]


class ConsecutivePersistence:
    """Fires once a condition has been True for `n` consecutive updates; stays
    fired while it remains True; resets on False."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self._n = n
        self._streak = 0

    def update(self, condition: bool) -> bool:
        if condition:
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self._n

    @property
    def streak(self) -> int:
        return self._streak
