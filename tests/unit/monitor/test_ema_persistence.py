import math

import pytest

from env_rl.monitor.ema import EMA, ConsecutivePersistence, smooth


def test_ema_converges_to_constant_input() -> None:
    ema = EMA(alpha=0.1)
    for _ in range(100):
        ema.update(10.0)
    assert math.isclose(ema.value, 10.0, rel_tol=1e-3)


def test_ema_initial_value_is_first_update() -> None:
    ema = EMA(alpha=0.3)
    ema.update(5.0)
    assert ema.value == 5.0


def test_ema_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        EMA(alpha=0.0)
    with pytest.raises(ValueError):
        EMA(alpha=1.5)


def test_smooth_returns_trace_same_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    trace = smooth(values, alpha=0.5)
    assert len(trace) == len(values)
    assert trace[0] == 1.0  # first value seeds


def test_smooth_empty_returns_empty() -> None:
    assert smooth([], alpha=0.5) == []


def test_consecutive_persistence_fires_after_n_consecutive_true() -> None:
    p = ConsecutivePersistence(n=3)
    assert not p.update(True)
    assert not p.update(True)
    assert p.update(True)  # third consecutive


def test_consecutive_persistence_resets_on_false() -> None:
    p = ConsecutivePersistence(n=3)
    p.update(True)
    p.update(True)
    p.update(False)  # reset
    assert not p.update(True)
    assert not p.update(True)
    assert p.update(True)


def test_consecutive_persistence_stays_fired_while_true() -> None:
    p = ConsecutivePersistence(n=2)
    p.update(True)
    assert p.update(True)
    assert p.update(True)  # still firing
    assert not p.update(False)
