"""Canonical rule evaluators for the 7-rule playbook.

Each rule consumes the full per-epoch metrics history and returns True if the
rule is currently firing at the latest epoch. EMA smoothing and 3-consecutive-
epoch persistence are applied uniformly via helpers in :mod:`env_rl.monitor.ema`.

The LLM never computes these signals — they are measured directly on the live
model by the monitor's hooks (see M3). The rule functions only *interpret* the
history; the thresholds come from the Hydra config (``conf/monitor/default.yaml``).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from env_rl.monitor.ema import smooth

MetricsHistory = Sequence[Mapping[str, Any]]
Config = Mapping[str, Any]


def _ema_tail_above(values: Sequence[float], alpha: float, n: int, thresh: float) -> bool:
    """EMA-smoothed `values`; returns True iff the last `n` EMA values > thresh."""
    if len(values) < n:
        return False
    trace = smooth(values, alpha)
    return all(v > thresh for v in trace[-n:])


def _ema_tail_below(values: Sequence[float], alpha: float, n: int, thresh: float) -> bool:
    if len(values) < n:
        return False
    trace = smooth(values, alpha)
    return all(v < thresh for v in trace[-n:])


def _ema_tail_outside(
    values: Sequence[float], alpha: float, n: int, low: float, high: float
) -> bool:
    if len(values) < n:
        return False
    trace = smooth(values, alpha)
    return all((v < low) or (v > high) for v in trace[-n:])


def _alpha(config: Config) -> float:
    return float(config.get("ema", {}).get("alpha", 0.1))


def _n(config: Config) -> int:
    return int(config.get("persistence", {}).get("consecutive_epochs", 3))


def _signal(history: MetricsHistory, key: str) -> list[float]:
    return [float(h[key]) for h in history if key in h]


# ---------------------------------------------------------------------------
# R1 — Learning Rate
# ---------------------------------------------------------------------------

def rule_r1(history: MetricsHistory, config: Config) -> bool:
    cfg = config["rules"]["r1_learning_rate"]
    alpha, n = _alpha(config), _n(config)

    ratios = _signal(history, "update_to_param_ratio")
    if _ema_tail_above(ratios, alpha, n, float(cfg["update_ratio_high"])):
        return True
    if _ema_tail_below(ratios, alpha, n, float(cfg["update_ratio_low"])):
        return True

    # Plateau branch: val_loss has not improved for `plateau_patience` epochs.
    patience = int(cfg["plateau_patience"])
    losses = _signal(history, "val_loss")
    if len(losses) >= patience + 1:
        window = losses[-(patience + 1) :]
        best_before_tail = min(window[:-patience])
        if all(v >= best_before_tail for v in window[-patience:]):
            return True
    return False


# ---------------------------------------------------------------------------
# R2 — Batch Size
# ---------------------------------------------------------------------------

def rule_r2(history: MetricsHistory, config: Config) -> bool:
    cfg = config["rules"]["r2_batch_size"]
    low, high = map(float, cfg["grad_noise_scale_band"])
    values = _signal(history, "grad_noise_scale")
    return _ema_tail_outside(values, _alpha(config), _n(config), low, high)


# ---------------------------------------------------------------------------
# R3 — Early Stopping
# ---------------------------------------------------------------------------

def rule_r3(history: MetricsHistory, config: Config) -> bool:
    cfg = config["rules"]["r3_early_stopping"]
    patience = int(cfg["patience"])
    min_delta = float(cfg["min_delta"])
    losses = _signal(history, "val_loss")
    if len(losses) < patience + 1:
        return False
    best_before = min(losses[: -patience])
    tail = losses[-patience:]
    return all(v >= best_before - min_delta for v in tail)


# ---------------------------------------------------------------------------
# R4 — Depth (capacity saturation)
# ---------------------------------------------------------------------------

def rule_r4(history: MetricsHistory, config: Config) -> bool:
    cfg = config["rules"]["r4_depth"]
    gap = float(cfg["saturation_gap"])
    accs = _signal(history, "train_acc")
    n = _n(config)
    if len(accs) < n + 1:
        return False
    diffs = [accs[i] - accs[i - 1] for i in range(-n, 0)]
    return all(abs(d) < gap for d in diffs)


# ---------------------------------------------------------------------------
# R5 — Activations (dead ReLU)
# ---------------------------------------------------------------------------

def rule_r5(history: MetricsHistory, config: Config) -> bool:
    cfg = config["rules"]["r5_activations"]
    values = _signal(history, "dead_relu_fraction")
    return _ema_tail_above(values, _alpha(config), _n(config), float(cfg["dead_relu_fraction"]))


# ---------------------------------------------------------------------------
# R6 — Vanishing Gradients
# ---------------------------------------------------------------------------

def rule_r6(history: MetricsHistory, config: Config) -> bool:
    cfg = config["rules"]["r6_vanishing_gradients"]
    values = _signal(history, "min_layer_grad_norm")
    return _ema_tail_below(
        values, _alpha(config), _n(config), float(cfg["min_layer_grad_norm"])
    )


# ---------------------------------------------------------------------------
# R7 — Exploding Gradients
# ---------------------------------------------------------------------------

def rule_r7(history: MetricsHistory, config: Config) -> bool:
    # Immediate fire on any NaN/Inf loss in history's tail.
    if history and _has_nan_or_inf(history[-1]):
        return True
    cfg = config["rules"]["r7_exploding_gradients"]
    values = _signal(history, "max_layer_grad_norm")
    return _ema_tail_above(
        values, _alpha(config), _n(config), float(cfg["max_layer_grad_norm"])
    )


def _has_nan_or_inf(metrics: Mapping[str, Any]) -> bool:
    for k in ("train_loss", "val_loss"):
        v = metrics.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            return True
    return bool(metrics.get("has_nan_or_inf", False))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

RULES = {
    "R1": rule_r1,
    "R2": rule_r2,
    "R3": rule_r3,
    "R4": rule_r4,
    "R5": rule_r5,
    "R6": rule_r6,
    "R7": rule_r7,
}


def evaluate_rules(history: MetricsHistory, config: Config) -> dict[str, bool]:
    """Single source of truth: returns {R1..R7: bool} for the latest epoch."""
    return {name: fn(history, config) for name, fn in RULES.items()}
