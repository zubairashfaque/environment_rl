from env_rl.monitor.rules import rule_r1
from tests._helpers import build_history


def test_r1_not_fired_on_clean_signal(monitor_config) -> None:
    # stable ratio in-band, val_loss improving — nothing should trip
    history = build_history(
        update_to_param_ratio=[1e-3] * 10,
        val_loss=[2.0 - 0.1 * i for i in range(10)],
    )
    assert rule_r1(history, monitor_config) is False


def test_r1_fires_when_update_ratio_high_for_3_consecutive_epochs(monitor_config) -> None:
    history = build_history(
        update_to_param_ratio=[1e-3] * 5 + [5e-2, 5e-2, 5e-2],
        val_loss=[2.0] * 8,
    )
    assert rule_r1(history, monitor_config) is True


def test_r1_fires_when_update_ratio_low_for_3_consecutive_epochs(monitor_config) -> None:
    history = build_history(
        update_to_param_ratio=[1e-3] * 5 + [1e-6, 1e-6, 1e-6],
        val_loss=[2.0] * 8,
    )
    assert rule_r1(history, monitor_config) is True


def test_r1_fires_on_val_loss_plateau(monitor_config) -> None:
    # best val_loss was 0.9 at epoch 3; plateau at 1.0 for the last 3 epochs
    history = build_history(
        update_to_param_ratio=[1e-3] * 7,
        val_loss=[2.0, 1.5, 1.0, 0.9, 1.0, 1.0, 1.0],
    )
    assert rule_r1(history, monitor_config) is True


def test_r1_clears_after_ratio_returns_to_band(monitor_config) -> None:
    history = build_history(
        update_to_param_ratio=[1e-3] * 5 + [5e-2, 5e-2, 5e-2] + [1e-3] * 8,
        val_loss=[2.0 - 0.1 * i for i in range(16)],
    )
    assert rule_r1(history, monitor_config) is False
