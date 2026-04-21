from env_rl.monitor.rules import rule_r4
from tests._helpers import build_history


def test_r4_not_fired_when_train_acc_improving(monitor_config) -> None:
    history = build_history(train_acc=[0.3 + 0.05 * i for i in range(10)])
    assert rule_r4(history, monitor_config) is False


def test_r4_fires_on_3_consecutive_flat_epochs(monitor_config) -> None:
    # train_acc gains < saturation_gap (0.02) for 3 consecutive epochs
    history = build_history(train_acc=[0.3, 0.5, 0.7, 0.8, 0.805, 0.810, 0.815])
    assert rule_r4(history, monitor_config) is True


def test_r4_clears_after_a_big_gain(monitor_config) -> None:
    history = build_history(
        train_acc=[0.3, 0.5, 0.7, 0.8, 0.805, 0.810, 0.815, 0.900]
    )
    # last 3 diffs: 0.815→0.900 (big), 0.810→0.815 (small), 0.805→0.810 (small)
    # since rule looks at the most recent n diffs, the tail includes a big jump → not firing
    assert rule_r4(history, monitor_config) is False
