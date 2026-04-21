from env_rl.monitor.rules import evaluate_rules
from tests._helpers import build_history


def test_evaluate_rules_returns_all_seven_keys(monitor_config) -> None:
    history = build_history(val_loss=[2.0, 1.5])
    evals = evaluate_rules(history, monitor_config)
    assert set(evals.keys()) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}


def test_evaluate_rules_returns_bools(monitor_config) -> None:
    history = build_history(val_loss=[2.0, 1.5])
    evals = evaluate_rules(history, monitor_config)
    assert all(isinstance(v, bool) for v in evals.values())


def test_evaluate_rules_multiple_rules_can_fire(monitor_config) -> None:
    # both R6 (min grad too low) and R7 (max grad too high) in same history
    history = build_history(
        min_layer_grad_norm=[1e-7] * 10,
        max_layer_grad_norm=[50.0] * 10,
    )
    evals = evaluate_rules(history, monitor_config)
    assert evals["R6"] is True
    assert evals["R7"] is True
