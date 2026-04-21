"""Judge step 7: live diagnostic sanity check on the submitted model.

Loads the final ``best_model.pt`` weights, runs one forward/backward pass on
a fixed batch drawn from the train split, measures the per-layer gradient
norms directly, and compares them to the last ``epoch`` record in
``metrics_log.jsonl``. A gross mismatch implies a fabricated trajectory —
hard fail.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_rl.judge.deliverables import HardFail


def measure_live_gradient_norms(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Run one fwd/bwd pass and return per-parameter grad norms."""
    was_training = model.training
    model.train()  # need training mode for BN stats / dropout to behave
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    x, y = batch
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    norms = {
        name: float(p.grad.norm().item())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    if was_training is False:
        model.eval()
    return norms


def compare_norms_within_tolerance(
    live_norms: dict[str, float],
    logged_per_layer: dict[str, float],
    *,
    tolerance: float = 0.30,
) -> None:
    """Raise HardFail if the max-layer norm differs from the logged value by
    more than ``tolerance`` (relative). This is a coarse sanity check — not a
    full trajectory replay.
    """
    if not live_norms:
        raise HardFail("no live gradients measured; model may be disconnected")

    live_max = max(live_norms.values())
    logged_max = max(logged_per_layer.values()) if logged_per_layer else None
    if logged_max is None:
        raise HardFail("logged metrics missing per_layer_grad_norm")

    if logged_max == 0 and live_max > 0:
        raise HardFail(
            "live gradient nonzero but logged per_layer_grad_norm was all zero"
        )
    if logged_max > 0:
        rel = abs(live_max - logged_max) / logged_max
        if rel > tolerance:
            raise HardFail(
                f"live/logged gradient norm disagreement: "
                f"live_max={live_max:.4g}, logged_max={logged_max:.4g}, rel_err={rel:.3f}"
            )


def run_live_diagnostic(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    final_epoch_metrics: dict,
    *,
    tolerance: float = 0.30,
    max_batches: int = 3,
) -> dict[str, float]:
    """Measure average live grad norms over a few fixed batches; compare to
    the logged final-epoch snapshot.
    """
    accum: dict[str, list[float]] = {}
    count = 0
    for batch in batches:
        if count >= max_batches:
            break
        per = measure_live_gradient_norms(model, batch)
        for name, v in per.items():
            accum.setdefault(name, []).append(v)
        count += 1
    if count == 0:
        raise HardFail("no batches provided for live diagnostic")

    live = {n: sum(v) / len(v) for n, v in accum.items()}
    logged_per_layer = dict(final_epoch_metrics.get("per_layer_grad_norm", {}))
    compare_norms_within_tolerance(live, logged_per_layer, tolerance=tolerance)
    return live
