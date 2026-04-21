"""Run the reference training agent end-to-end.

Usage::

    poetry run python examples/run_reference_agent.py --synthetic --epochs 3

With ``--synthetic`` the script trains on tiny random tensors (no CIFAR-10
download required) — useful for smoke tests and CI. Without it, the CIFAR-10
loaders from :mod:`env_rl.data.loaders` are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env_rl.agent.reference_run import (
    ReferenceRunConfig,
    _make_synthetic_loader,
    run_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--judge-logs", default="./judge_logs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--data-dir", default="./data/cifar10")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    cfg = ReferenceRunConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        workspace=args.workspace,
    )

    if args.synthetic:
        train_loader = _make_synthetic_loader(n_batches=8, batch_size=args.batch_size, seed=cfg.seed)
        val_loader = _make_synthetic_loader(n_batches=2, batch_size=args.batch_size, seed=cfg.seed + 1)
    else:
        from env_rl.data.loaders import make_loader
        train_loader = list(make_loader("train", batch_size=args.batch_size, data_dir=args.data_dir, manifest_path=args.manifest))
        val_loader = list(make_loader("val", batch_size=args.batch_size, data_dir=args.data_dir, manifest_path=args.manifest))

    monitor_config = {
        "log_dir": args.judge_logs,
        "root_hash": "0" * 64,
        "ema": {"alpha": 0.1},
        "persistence": {"consecutive_epochs": 3},
        "rules": {
            "r1_learning_rate": {"update_ratio_high": 1e-2, "update_ratio_low": 1e-4, "plateau_patience": 3},
            "r2_batch_size": {"grad_noise_scale_band": [50.0, 5000.0]},
            "r3_early_stopping": {"patience": 5, "min_delta": 1e-3},
            "r4_depth": {"saturation_gap": 0.02},
            "r5_activations": {"dead_relu_fraction": 0.40},
            "r6_vanishing_gradients": {"min_layer_grad_norm": 1e-5},
            "r7_exploding_gradients": {"max_layer_grad_norm": 10.0},
        },
    }
    Path(args.judge_logs).mkdir(parents=True, exist_ok=True)
    summary = run_reference(cfg, train_loader, val_loader, monitor_config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
