from pathlib import Path

import pytest


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "judge_logs"
    d.mkdir()
    return d


@pytest.fixture
def root_hash() -> str:
    return "0" * 64


@pytest.fixture
def monitor_config() -> dict:
    """Mirror of conf/monitor/default.yaml for unit tests."""
    return {
        "ema": {"alpha": 0.1},
        "persistence": {"consecutive_epochs": 3},
        "rules": {
            "r1_learning_rate": {
                "update_ratio_high": 1.0e-2,
                "update_ratio_low": 1.0e-4,
                "plateau_patience": 3,
            },
            "r2_batch_size": {"grad_noise_scale_band": [50.0, 5000.0]},
            "r3_early_stopping": {"patience": 5, "min_delta": 1.0e-3},
            "r4_depth": {"saturation_gap": 0.02},
            "r5_activations": {"dead_relu_fraction": 0.40},
            "r6_vanishing_gradients": {"min_layer_grad_norm": 1.0e-5},
            "r7_exploding_gradients": {"max_layer_grad_norm": 10.0},
        },
    }


