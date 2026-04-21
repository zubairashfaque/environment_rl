from env_rl.monitor.rules import rule_r5
from tests._helpers import build_history


def test_r5_not_fired_with_healthy_activations(monitor_config) -> None:
    history = build_history(dead_relu_fraction=[0.05] * 10)
    assert rule_r5(history, monitor_config) is False


def test_r5_fires_when_dead_fraction_elevated_for_3_epochs(monitor_config) -> None:
    history = build_history(dead_relu_fraction=[0.05] * 5 + [0.7] * 20)
    assert rule_r5(history, monitor_config) is True


def test_r5_does_not_fire_on_single_spike(monitor_config) -> None:
    history = build_history(dead_relu_fraction=[0.05] * 5 + [0.95] + [0.05] * 5)
    assert rule_r5(history, monitor_config) is False


def test_r5_clears_after_activations_recover(monitor_config) -> None:
    history = build_history(
        dead_relu_fraction=[0.7] * 4 + [0.05] * 30
    )
    assert rule_r5(history, monitor_config) is False
