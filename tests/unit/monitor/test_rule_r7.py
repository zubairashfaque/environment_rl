import math

from env_rl.monitor.rules import rule_r7
from tests._helpers import build_history


def test_r7_not_fired_with_healthy_grads(monitor_config) -> None:
    history = build_history(max_layer_grad_norm=[2.0] * 10)
    assert rule_r7(history, monitor_config) is False


def test_r7_fires_when_max_grad_above_threshold_for_3_epochs(monitor_config) -> None:
    history = build_history(max_layer_grad_norm=[2.0] * 5 + [50.0] * 20)
    assert rule_r7(history, monitor_config) is True


def test_r7_fires_immediately_on_nan_loss(monitor_config) -> None:
    history = build_history(
        max_layer_grad_norm=[2.0, 2.0],
        train_loss=[1.0, math.nan],
    )
    assert rule_r7(history, monitor_config) is True


def test_r7_fires_immediately_on_inf_loss(monitor_config) -> None:
    history = build_history(
        max_layer_grad_norm=[2.0, 2.0],
        val_loss=[1.0, math.inf],
    )
    assert rule_r7(history, monitor_config) is True


def test_r7_clears_after_grads_shrink(monitor_config) -> None:
    history = build_history(
        max_layer_grad_norm=[50.0] * 4 + [2.0] * 30,
    )
    assert rule_r7(history, monitor_config) is False
