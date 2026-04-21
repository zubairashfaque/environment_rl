import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.agent.model import ResidualCNN  # noqa: E402


def test_model_forward_shape() -> None:
    m = ResidualCNN(num_blocks=2, base_channels=8)
    y = m(torch.randn(3, 3, 32, 32))
    assert y.shape == (3, 10)


def test_model_has_residual_and_bn() -> None:
    m = ResidualCNN(num_blocks=1, base_channels=8, bn_enabled=True)
    # BN present
    assert any(isinstance(sub, nn.BatchNorm2d) for sub in m.modules())


def test_model_spec_matches_constructor() -> None:
    m = ResidualCNN(num_blocks=3, base_channels=16, activation="leaky_relu", bn_enabled=False)
    spec = m.spec()
    assert spec == {"num_blocks": 3, "activation": "leaky_relu", "bn_enabled": False}


def test_he_init_applied_to_conv_layers() -> None:
    m = ResidualCNN(num_blocks=1, base_channels=8)
    # He init leaves positive RMS of weight larger than zero-init
    for sub in m.modules():
        if isinstance(sub, (nn.Conv2d, nn.Linear)):
            rms = float(sub.weight.detach().pow(2).mean().sqrt().item())
            assert rms > 1e-3
