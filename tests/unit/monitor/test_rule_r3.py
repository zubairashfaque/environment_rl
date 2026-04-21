from env_rl.monitor.rules import rule_r3
from tests._helpers import build_history


def test_r3_not_fired_while_val_loss_improving(monitor_config) -> None:
    history = build_history(val_loss=[2.0 - 0.1 * i for i in range(10)])
    assert rule_r3(history, monitor_config) is False


def test_r3_fires_after_patience_epochs_without_improvement(monitor_config) -> None:
    # best = 1.0 at epoch 4; 5 flat epochs after
    history = build_history(val_loss=[2.0, 1.8, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert rule_r3(history, monitor_config) is True


def test_r3_not_fired_if_improvement_within_patience(monitor_config) -> None:
    history = build_history(val_loss=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    assert rule_r3(history, monitor_config) is False


def test_r3_respects_min_delta(monitor_config) -> None:
    # improvement smaller than min_delta (1e-3) does not count
    history = build_history(
        val_loss=[1.0, 0.9995, 0.9994, 0.9993, 0.9992, 0.9991, 0.9990]
    )
    assert rule_r3(history, monitor_config) is True
