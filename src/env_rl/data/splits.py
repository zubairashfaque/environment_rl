"""Deterministic CIFAR-10 train/val split under a fixed seed.

The manifest written by :func:`build_splits` defines which samples of the
50000-image ``CIFAR10(train=True)`` partition go to train vs val. The held-out
test split is the full 10000-image ``CIFAR10(train=False)`` partition — it is
not indexed here because workspace code is not allowed to reach it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

TRAINVAL_SIZE = 50_000
TEST_SIZE = 10_000
DEFAULT_VAL_FRAC = 0.10


def build_splits(
    seed: int = 42,
    val_frac: float = DEFAULT_VAL_FRAC,
    trainval_size: int = TRAINVAL_SIZE,
) -> dict[str, Any]:
    """Return a deterministic split manifest keyed on ``seed``.

    Shape::
        {
          "seed": 42,
          "trainval_size": 50000,
          "val_frac": 0.1,
          "train": [indices into CIFAR10(train=True)],
          "val":   [indices into CIFAR10(train=True)],
          "test_source": "torchvision_test",
        }

    The ``train`` and ``val`` lists are disjoint and together cover
    ``range(trainval_size)`` exactly. The test split is not enumerated.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    rng = np.random.default_rng(seed)
    indices = np.arange(trainval_size)
    rng.shuffle(indices)

    n_val = int(round(val_frac * trainval_size))
    val = sorted(indices[:n_val].tolist())
    train = sorted(indices[n_val:].tolist())

    return {
        "seed": seed,
        "trainval_size": trainval_size,
        "val_frac": val_frac,
        "train": train,
        "val": val,
        "test_source": "torchvision_test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write deterministic splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_splits(seed=args.seed, val_frac=args.val_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out} "
          f"(train={len(manifest['train'])}, val={len(manifest['val'])})")


if __name__ == "__main__":
    main()
