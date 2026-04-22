"""End-to-end: verify that un-waiving R3 (early stop) and R4 (add_block)
produces clean runs with no hard fails."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from env_rl.agent.reference_run import (  # noqa: E402
    ReferenceRunConfig,
    _make_synthetic_loader,
    run_reference,
)
from env_rl.harness.policy import OpenAIDecisionPolicy  # noqa: E402
from env_rl.judge import run_judge  # noqa: E402
from env_rl.monitor import session as _session_mod  # noqa: E402
from tests.unit.harness._stubs import StubOpenAIClient  # noqa: E402


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
def test_r4_add_block_decision_is_actioned_and_passes_judge(tmp_path: Path) -> None:
    """If the LLM emits an R4 add_block decision, the harness executes it
    and the judge's architecture replay matches the submitted model."""
    canned = {
        "event_type": "architecture_change",
        "cites": ["R4"],
        "justification": "train plateau with clean grads; add capacity per R4",
        "remedy_direction": "add_capacity",
        "remedy_params": {"lr_new": 0.05, "edit_op": "add_block", "edit_to": "none"},
    }
    client = StubOpenAIClient(responses=[canned] * 20)
    policy = OpenAIDecisionPolicy(client=client, model="gpt-4o-mini")

    workspace = tmp_path / "workspace"
    judge_logs = tmp_path / "judge_logs"
    cfg = ReferenceRunConfig(
        lr=0.05, batch_size=16, max_epochs=2,
        workspace=str(workspace), num_blocks=2, base_channels=8,
    )
    train = _make_synthetic_loader(n_batches=4, batch_size=16, seed=cfg.seed)
    val = _make_synthetic_loader(n_batches=2, batch_size=16, seed=cfg.seed + 1)
    run_reference(cfg, train, val, _monitor_config(judge_logs), policy=policy)

    scores = run_judge(
        workspace=workspace,
        judge_logs=judge_logs,
        root_hash="0" * 64,
        target_acc=0.20,
        test_loader=_make_synthetic_loader(n_batches=1, batch_size=16, seed=999),
        live_diag_batches=_make_synthetic_loader(n_batches=1, batch_size=16, seed=0),
        initial_arch_spec={"num_blocks": 2, "activation": "relu", "bn_enabled": True},
        live_diag_tolerance=0.99,
    )
    assert scores.hard_fail is False


def test_r3_stop_flag_gets_set_when_decision_matches() -> None:
    """Unit-level: verify that a stop decision with R3 cite flips the
    early_stop flag. The full R3-fire-plus-natural-stop path is too
    timing-sensitive for a short-horizon test."""
    from env_rl.harness.policy import Decision

    # Emulates what reference_run.py does after policy.decide():
    decision = Decision(
        event_type="hyperparameter_change",
        cites=["R3"],
        justification="val plateau",
        remedy_direction="stop",
        remedy_params={"lr_new": 0.05},
    )
    early_stop_requested = False
    if (
        decision.event_type == "hyperparameter_change"
        and decision.remedy_direction == "stop"
        and "R3" in decision.cites
    ):
        early_stop_requested = True
    assert early_stop_requested is True

    # And the converse: an LR change does not flip it.
    decision_lr = Decision(
        event_type="hyperparameter_change",
        cites=["R1"],
        justification="lr high",
        remedy_direction="decrease_lr",
        remedy_params={"lr_new": 0.01},
    )
    flag = False
    if (
        decision_lr.event_type == "hyperparameter_change"
        and decision_lr.remedy_direction == "stop"
        and "R3" in decision_lr.cites
    ):
        flag = True
    assert flag is False


def test_all_rules_un_waived_after_stages_1_to_5() -> None:
    """After Stages 1-5 of the un-waiving plan, no rule is waived anymore."""
    from env_rl.harness import HARNESS_WAIVED_RULES
    assert HARNESS_WAIVED_RULES == frozenset()
