import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from env_rl.agent.reference_run import (  # noqa: E402
    ReferenceRunConfig,
    _make_synthetic_loader,
    run_reference,
)
from env_rl.judge import run_judge  # noqa: E402
from env_rl.monitor import session as _session_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_session():
    _session_mod._reset_for_tests()
    yield
    _session_mod._reset_for_tests()


def _monitor_config(log_dir: Path) -> dict:
    return {
        "log_dir": str(log_dir),
        "root_hash": "0" * 64,
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


@pytest.mark.integration
def test_full_reference_run_then_judge(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    judge_logs = tmp_path / "judge_logs"
    cfg = ReferenceRunConfig(
        lr=0.05,
        batch_size=16,
        max_epochs=2,
        workspace=str(workspace),
        num_blocks=2,
        base_channels=8,
    )
    train_loader = _make_synthetic_loader(n_batches=4, batch_size=16, seed=cfg.seed)
    val_loader = _make_synthetic_loader(n_batches=2, batch_size=16, seed=cfg.seed + 1)

    summary = run_reference(cfg, train_loader, val_loader, _monitor_config(judge_logs))
    assert "best_val_acc" in summary

    # All three deliverables exist
    for name in ("model.py", "best_model.pt", "run_config.json"):
        assert (workspace / name).is_file()
    # All three logs exist
    for name in ("metrics_log.jsonl", "decision_log.jsonl", "rule_evaluations.jsonl"):
        assert (judge_logs / name).is_file()

    test_loader = _make_synthetic_loader(n_batches=2, batch_size=16, seed=999)
    live_batches = _make_synthetic_loader(n_batches=1, batch_size=16, seed=0)
    scores = run_judge(
        workspace=workspace,
        judge_logs=judge_logs,
        root_hash="0" * 64,
        target_acc=0.20,  # synthetic data, low bar
        test_loader=test_loader,
        live_diag_batches=live_batches,
        initial_arch_spec={"num_blocks": 2, "activation": "relu", "bn_enabled": True},
        live_diag_tolerance=0.99,  # very loose for synthetic data
    )

    # Core integration expectations
    assert scores.hard_fail is False
    assert scores.process_score >= 0.9
    # Accuracy on random labels will be ~0.1 for 10-way; target 0.2 → monotone below 1.
    assert 0.0 <= scores.accuracy_score <= 1.0
