"""Judge step 10: test-set evaluation.

Loads the held-out CIFAR-10 test split using the SAME manifest the LLM was
given, runs inference with the loaded model in ``eval()`` mode, and returns
accuracy.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def evaluate_accuracy(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Top-1 accuracy over ``loader``. Deterministic for a fixed loader and model."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    if total == 0:
        return 0.0
    return correct / total
