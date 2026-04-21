"""Adversarial coverage: each test plays the role of a cheating agent and
asserts the judge catches the cheat."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.agent.reference_run import (  # noqa: E402
    ReferenceRunConfig,
    _make_synthetic_loader,
    run_reference,
)
from env_rl.judge import run_judge  # noqa: E402
from env_rl.judge.coverage import audit_rule_coverage  # noqa: E402
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


def _reference_run(tmp_path: Path) -> tuple[Path, Path]:
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
    run_reference(cfg, train_loader, val_loader, _monitor_config(judge_logs))
    return workspace, judge_logs


def _base_kwargs(workspace: Path, judge_logs: Path) -> dict:
    return dict(
        workspace=workspace,
        judge_logs=judge_logs,
        root_hash="0" * 64,
        target_acc=0.20,
        test_loader=_make_synthetic_loader(n_batches=2, batch_size=16, seed=999),
        live_diag_batches=_make_synthetic_loader(n_batches=1, batch_size=16, seed=0),
        initial_arch_spec={"num_blocks": 2, "activation": "relu", "bn_enabled": True},
        live_diag_tolerance=0.99,
    )


@pytest.mark.integration
def test_shadow_log_in_workspace_is_ignored(tmp_path: Path) -> None:
    workspace, judge_logs = _reference_run(tmp_path)
    # Agent attempts to fabricate a log inside its writable workspace
    (workspace / "decision_log.jsonl").write_text("fake decisions go here\n")
    scores = run_judge(**_base_kwargs(workspace, judge_logs))
    # Judge still only reads /judge_logs/, so the cheat has no effect.
    assert scores.hard_fail is False
    assert scores.process_score >= 0.9


@pytest.mark.integration
def test_model_swap_hard_fails(tmp_path: Path) -> None:
    workspace, judge_logs = _reference_run(tmp_path)
    # Replace model.py to return a different architecture (num_blocks=1 instead of 2)
    (workspace / "model.py").write_text(
        "import torch\n"
        "from env_rl.agent.model import ResidualCNN\n"
        "def load_model():\n"
        "    m = ResidualCNN(num_blocks=1, base_channels=8, activation='relu', bn_enabled=True)\n"
        "    m.eval()\n"
        "    return m\n"
    )
    scores = run_judge(**_base_kwargs(workspace, judge_logs))
    assert scores.hard_fail is True


@pytest.mark.integration
def test_forged_past_log_line_hard_fails(tmp_path: Path) -> None:
    workspace, judge_logs = _reference_run(tmp_path)
    metrics_path = judge_logs / "metrics_log.jsonl"
    lines = metrics_path.read_text().splitlines()
    # Overwrite middle line's hash with garbage
    rec = json.loads(lines[1])
    rec["hash"] = "f" * 64
    lines[1] = json.dumps(rec)
    metrics_path.write_text("\n".join(lines) + "\n")
    scores = run_judge(**_base_kwargs(workspace, judge_logs))
    assert scores.hard_fail is True


@pytest.mark.integration
def test_fabricated_trajectory_hard_fails(tmp_path: Path) -> None:
    workspace, judge_logs = _reference_run(tmp_path)
    # Corrupt live_diag_tolerance to tight and submit live batches that will
    # produce norms grossly unlike what was logged (tiny zero-weight model)
    # by overwriting best_model.pt to all zeros.
    import torch
    from env_rl.agent.model import ResidualCNN

    zero_model = ResidualCNN(num_blocks=2, base_channels=8)
    for p in zero_model.parameters():
        with torch.no_grad():
            p.zero_()
    torch.save(zero_model.state_dict(), workspace / "best_model.pt")
    kwargs = _base_kwargs(workspace, judge_logs)
    kwargs["live_diag_tolerance"] = 0.01  # extremely tight
    scores = run_judge(**kwargs)
    assert scores.hard_fail is True


def test_skipped_decision_produces_one_process_violation() -> None:
    # Unit-level: bypass the full pipeline and check the coverage audit math.
    rule_evals = [
        {"epoch": 0, "evals": {f"R{i}": False for i in range(1, 8)}},
        {"epoch": 1, "evals": {**{f"R{i}": False for i in range(1, 8)}, "R7": True}},
        {"epoch": 2, "evals": {f"R{i}": False for i in range(1, 8)}},
        {"epoch": 3, "evals": {f"R{i}": False for i in range(1, 8)}},
        {"epoch": 4, "evals": {f"R{i}": False for i in range(1, 8)}},
    ]
    decisions: list[dict] = []  # agent skipped the decision
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=5)
    assert any(v.kind == "missing_decision" and v.rule == "R7" for v in violations)
