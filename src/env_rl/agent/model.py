"""Residual CNN meeting the model contract.

Requirements from the design doc:
  - ≥1 residual connection
  - ≥1 BatchNorm layer
  - He init on conv and linear layers
  - spec() method so the judge's architecture replay has a canonical form
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def he_init_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "gelu": nn.GELU,
}


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        activation: str,
        bn_enabled: bool,
    ) -> None:
        super().__init__()
        self.bn_enabled = bn_enabled
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not bn_enabled)
        self.bn1 = nn.BatchNorm2d(channels) if bn_enabled else nn.Identity()
        self.act1 = _ACTIVATIONS[activation]()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not bn_enabled)
        self.bn2 = nn.BatchNorm2d(channels) if bn_enabled else nn.Identity()
        self.act2 = _ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(x + y)  # residual connection


class ResidualCNN(nn.Module):
    """Small residual CNN for CIFAR-10.

    The forward arg shape is ``[N, 3, 32, 32]``; output is logits ``[N, 10]``.
    """

    def __init__(
        self,
        *,
        num_blocks: int = 6,
        base_channels: int = 32,
        num_classes: int = 10,
        activation: str = "relu",
        bn_enabled: bool = True,
    ) -> None:
        super().__init__()
        self._num_blocks = num_blocks
        self._activation = activation
        self._bn_enabled = bn_enabled

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=not bn_enabled),
            nn.BatchNorm2d(base_channels) if bn_enabled else nn.Identity(),
            _ACTIVATIONS[activation](),
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock(base_channels, activation=activation, bn_enabled=bn_enabled)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(base_channels, num_classes)
        he_init_(self)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def spec(self) -> dict:
        return {
            "num_blocks": self._num_blocks,
            "activation": self._activation,
            "bn_enabled": self._bn_enabled,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        for b in self.blocks:
            y = b(y)
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        return self.head(y)
