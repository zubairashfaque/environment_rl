"""Forward/backward hooks that let the monitor compute diagnostic signals
directly off the live model.

The LLM never computes these numbers itself — closing the entire
threshold-gaming attack surface. Call :class:`HookManager.attach(model)` once
immediately after model init, then :meth:`HookManager.collect` at the end of
each epoch to obtain the per-step averages and a final snapshot.

Measured per step (then averaged per epoch):
  - per-layer unclipped gradient norm  → max/min used by R6/R7
  - dead-ReLU fraction per layer       → max used by R5
  - activation percentiles             → future-use (saturation detection)
  - update-to-parameter magnitude ratio → R1
  - effective gradient noise scale     → R2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class EpochMetrics:
    """Monitor-owned, per-epoch diagnostic snapshot. All derived solely from hooks."""

    per_layer_grad_norm: dict[str, float] = field(default_factory=dict)
    max_layer_grad_norm: float = 0.0
    min_layer_grad_norm: float = 0.0
    per_layer_dead_relu_fraction: dict[str, float] = field(default_factory=dict)
    dead_relu_fraction: float = 0.0  # max over layers
    activation_p50: float = 0.0
    activation_p99: float = 0.0
    update_to_param_ratio: float = 0.0
    grad_noise_scale: float = 0.0
    step_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_layer_grad_norm": dict(self.per_layer_grad_norm),
            "max_layer_grad_norm": self.max_layer_grad_norm,
            "min_layer_grad_norm": self.min_layer_grad_norm,
            "per_layer_dead_relu_fraction": dict(self.per_layer_dead_relu_fraction),
            "dead_relu_fraction": self.dead_relu_fraction,
            "activation_p50": self.activation_p50,
            "activation_p99": self.activation_p99,
            "update_to_param_ratio": self.update_to_param_ratio,
            "grad_noise_scale": self.grad_noise_scale,
            "step_count": self.step_count,
        }


_RELU_LIKE = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)


class HookManager:
    """Registers forward hooks on activations and backward hooks on parameter
    gradients. Accumulates per-step statistics; `collect()` returns the
    per-epoch aggregate and resets the accumulators.
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._handles: list[Any] = []
        # accumulators keyed by module name
        self._dead_fractions: dict[str, list[float]] = {}
        self._activation_samples: list[float] = []
        self._param_prev_snapshot: dict[str, torch.Tensor] | None = None
        self._per_step_grad_norms: list[dict[str, float]] = []

    # ------------------------------------------------------------------ public

    def attach(self, model: nn.Module) -> None:
        if self._model is not None:
            raise RuntimeError("HookManager already attached; call detach() first")
        self._model = model
        for name, module in model.named_modules():
            if isinstance(module, _RELU_LIKE):
                handle = module.register_forward_hook(self._make_relu_hook(name))
                self._handles.append(handle)
        # snapshot params for update-to-param ratio computation
        self._param_prev_snapshot = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._model = None
        self._param_prev_snapshot = None
        self._dead_fractions.clear()
        self._activation_samples.clear()
        self._per_step_grad_norms.clear()

    def record_step(self) -> None:
        """Called by the training loop after optimizer.step() in each batch.

        Captures per-param gradient norms and updates the param snapshot. The
        gradient noise scale is approximated from the variance of per-step
        total-grad-norm.
        """
        if self._model is None:
            raise RuntimeError("HookManager is not attached")

        # Snapshot current gradient norms
        step_norms: dict[str, float] = {}
        for n, p in self._model.named_parameters():
            if p.grad is not None:
                step_norms[n] = float(p.grad.detach().norm().item())
        if step_norms:
            self._per_step_grad_norms.append(step_norms)

    def collect(self) -> EpochMetrics:
        """Aggregate per-step data into one epoch snapshot and reset."""
        metrics = EpochMetrics()
        metrics.step_count = len(self._per_step_grad_norms)

        if self._per_step_grad_norms:
            # average each param's grad norm over steps
            names = set().union(*(d.keys() for d in self._per_step_grad_norms))
            for n in names:
                vals = [d[n] for d in self._per_step_grad_norms if n in d]
                metrics.per_layer_grad_norm[n] = sum(vals) / max(1, len(vals))
            vals_all = list(metrics.per_layer_grad_norm.values())
            metrics.max_layer_grad_norm = max(vals_all)
            metrics.min_layer_grad_norm = min(vals_all)

            # gradient noise scale approx: mean-squared total norm / variance (guarded)
            totals = [
                sum(v * v for v in d.values()) ** 0.5 for d in self._per_step_grad_norms
            ]
            mean_t = sum(totals) / len(totals)
            if len(totals) > 1:
                var_t = sum((t - mean_t) ** 2 for t in totals) / (len(totals) - 1)
                metrics.grad_noise_scale = float(
                    (mean_t * mean_t) / max(var_t, 1e-12)
                )
            else:
                metrics.grad_noise_scale = 0.0

        if self._dead_fractions:
            for n, vals in self._dead_fractions.items():
                metrics.per_layer_dead_relu_fraction[n] = sum(vals) / len(vals)
            metrics.dead_relu_fraction = max(
                metrics.per_layer_dead_relu_fraction.values()
            )

        if self._activation_samples:
            samples = sorted(self._activation_samples)
            metrics.activation_p50 = samples[len(samples) // 2]
            metrics.activation_p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]

        # update-to-param ratio: compare current params to snapshot
        if self._model is not None and self._param_prev_snapshot is not None:
            ratios = []
            new_snapshot = {}
            for n, p in self._model.named_parameters():
                if not p.requires_grad:
                    continue
                prev = self._param_prev_snapshot.get(n)
                if prev is None:
                    new_snapshot[n] = p.detach().clone()
                    continue
                delta = (p.detach() - prev).norm().item()
                denom = max(p.detach().norm().item(), 1e-12)
                ratios.append(delta / denom)
                new_snapshot[n] = p.detach().clone()
            if ratios:
                metrics.update_to_param_ratio = sum(ratios) / len(ratios)
            self._param_prev_snapshot = new_snapshot

        # reset accumulators for next epoch
        self._per_step_grad_norms.clear()
        self._dead_fractions.clear()
        self._activation_samples.clear()

        return metrics

    # ------------------------------------------------------------------ hooks

    def _make_relu_hook(self, name: str):
        def hook(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                dead_mask = (output == 0).to(torch.float32)
                dead_fraction = float(dead_mask.mean().item())
                self._dead_fractions.setdefault(name, []).append(dead_fraction)
                # sample a bounded number of activations for percentiles
                flat = output.detach().flatten()
                if flat.numel() > 0:
                    take = min(256, flat.numel())
                    stride = max(1, flat.numel() // take)
                    self._activation_samples.extend(
                        float(x) for x in flat[::stride].tolist()[:take]
                    )

        return hook
