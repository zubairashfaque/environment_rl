from env_rl.monitor.rules import rule_r6
from tests._helpers import build_history


def test_r6_not_fired_with_healthy_grads(monitor_config) -> None:
    history = build_history(min_layer_grad_norm=[1e-2] * 10)
    assert rule_r6(history, monitor_config) is False


def test_r6_fires_when_min_grad_below_threshold_for_3_epochs(monitor_config) -> None:
    # signal below threshold from epoch 0; EMA seeds low and stays low
    history = build_history(min_layer_grad_norm=[1e-7] * 10)
    assert rule_r6(history, monitor_config) is True


def test_r6_clears_after_grads_recover(monitor_config) -> None:
    history = build_history(min_layer_grad_norm=[1e-7] * 4 + [1e-2] * 30)
    assert rule_r6(history, monitor_config) is False
