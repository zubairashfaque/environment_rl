"""Run the Iterative Self-Refine LLM agent (NOT reinforcement learning).

This script runs the training+judging pipeline N times with an OpenAI model
as the decision-maker. Between attempts, prior scores and violations are
injected into the next attempt's system prompt. Weights never change.

Example::

    export OPENAI_API_KEY=sk-...
    poetry run python examples/run_llm_agent.py \\
        --attempts 3 --epochs 3 --model gpt-4o-mini

See ``src/env_rl/harness/__init__.py`` for why this is explicitly NOT RL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from env_rl.agent.reference_run import (
    ReferenceRunConfig,
    _make_synthetic_loader,
    run_reference,
)
from env_rl.harness.iterative import run_iterative
from env_rl.judge import run_judge


def _monitor_config(log_dir: Path, root_hash: str = "0" * 64) -> dict:
    return {
        "log_dir": str(log_dir),
        "root_hash": root_hash,
        "ema": {"alpha": 0.1},
        "persistence": {"consecutive_epochs": 3},
        "rules": {
            "r1_learning_rate": {
                "update_ratio_high": 1e-2,
                "update_ratio_low": 1e-4,
                "plateau_patience": 3,
            },
            "r2_batch_size": {"grad_noise_scale_band": [50.0, 5000.0]},
            "r3_early_stopping": {"patience": 5, "min_delta": 1e-3},
            "r4_depth": {"saturation_gap": 0.02},
            "r5_activations": {"dead_relu_fraction": 0.40},
            "r6_vanishing_gradients": {"min_layer_grad_norm": 1e-5},
            "r7_exploding_gradients": {"max_layer_grad_norm": 10.0},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--target-acc", type=float, default=0.20)
    parser.add_argument("--base-dir", default="./llm_runs")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic random tensors (no CIFAR-10 download).",
    )
    parser.add_argument("--data-dir", default="./data/cifar10")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(2)

    # Lazy import so unit tests that don't hit this path don't need openai at import time.
    from openai import OpenAI

    client = OpenAI()

    base_dir = Path(args.base_dir)

    # Prepare loaders once per attempt - create fresh inside the closure so the
    # monitor's epoch counter starts at 0 each attempt.
    def run_one_attempt(policy, workspace, judge_logs):
        cfg = ReferenceRunConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            workspace=str(workspace),
            num_blocks=2,
            base_channels=16,
        )
        if args.synthetic:
            train_loader = _make_synthetic_loader(
                n_batches=8, batch_size=args.batch_size, seed=cfg.seed
            )
            val_loader = _make_synthetic_loader(
                n_batches=2, batch_size=args.batch_size, seed=cfg.seed + 1
            )
        else:
            from env_rl.data.loaders import make_loader

            train_loader = list(
                make_loader(
                    "train",
                    batch_size=args.batch_size,
                    data_dir=args.data_dir,
                    manifest_path=args.manifest,
                )
            )
            val_loader = list(
                make_loader(
                    "val",
                    batch_size=args.batch_size,
                    data_dir=args.data_dir,
                    manifest_path=args.manifest,
                )
            )
        run_reference(cfg, train_loader, val_loader, _monitor_config(judge_logs), policy=policy)
        test_loader = (
            _make_synthetic_loader(n_batches=2, batch_size=args.batch_size, seed=999)
            if args.synthetic
            else None
        )
        if test_loader is None:
            from env_rl.data.loaders import make_test_loader_judge_only

            test_loader = list(
                make_test_loader_judge_only(batch_size=args.batch_size, data_dir=args.data_dir)
            )
        live_batches = (
            _make_synthetic_loader(n_batches=1, batch_size=args.batch_size, seed=0)
            if args.synthetic
            else [next(iter(train_loader))]
        )
        return run_judge(
            workspace=workspace,
            judge_logs=judge_logs,
            root_hash="0" * 64,
            target_acc=args.target_acc,
            test_loader=test_loader,
            live_diag_batches=live_batches,
            initial_arch_spec={"num_blocks": 2, "activation": "relu", "bn_enabled": True},
            live_diag_tolerance=0.99 if args.synthetic else 0.50,
        )

    result = run_iterative(
        attempts=args.attempts,
        client=client,
        model_name=args.model,
        temperature=args.temperature,
        run_one_attempt=run_one_attempt,
        base_dir=base_dir,
    )

    from env_rl.harness import HARNESS_MODE

    print(
        json.dumps(
            {
                "mode": HARNESS_MODE,  # iterative_self_refine — NOT reinforcement learning
                "best_attempt": result.best.index,
                "best_scores": {
                    "accuracy_score": result.best.scores.accuracy_score,
                    "process_score": result.best.scores.process_score,
                    "test_accuracy": result.best.scores.test_accuracy,
                    "violations": result.best.scores.violations,
                    "total_decisions": result.best.scores.total_decisions,
                },
                "all_attempts": [
                    {
                        "index": a.index,
                        "accuracy_score": a.scores.accuracy_score,
                        "process_score": a.scores.process_score,
                        "violations": a.scores.violations,
                        "total_decisions": a.scores.total_decisions,
                    }
                    for a in result.all_attempts
                ],
                "base_dir": str(base_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
