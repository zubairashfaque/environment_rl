from env_rl.monitor.rules import rule_r2
from tests._helpers import build_history


def test_r2_not_fired_inside_band(monitor_config) -> None:
    history = build_history(grad_noise_scale=[500.0] * 10)
    assert rule_r2(history, monitor_config) is False


def test_r2_fires_when_gns_too_low_for_3_epochs(monitor_config) -> None:
    # signal starts below band; EMA seeds low and stays low
    history = build_history(grad_noise_scale=[5.0] * 10)
    assert rule_r2(history, monitor_config) is True


def test_r2_fires_when_gns_too_high_for_3_epochs(monitor_config) -> None:
    history = build_history(grad_noise_scale=[500.0] * 5 + [50_000.0] * 20)
    assert rule_r2(history, monitor_config) is True


def test_r2_clears_after_signal_returns_to_band(monitor_config) -> None:
    history = build_history(
        grad_noise_scale=[500.0] * 5 + [5.0] * 3 + [500.0] * 15
    )
    assert rule_r2(history, monitor_config) is False
