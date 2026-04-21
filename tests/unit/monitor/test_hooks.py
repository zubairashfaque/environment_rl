import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.monitor.hooks import HookManager  # noqa: E402


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def _fwd_bwd_step(model: nn.Module, optim: torch.optim.Optimizer, hm: HookManager) -> None:
    x = torch.randn(4, 8)
    y = torch.randint(0, 2, (4,))
    optim.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    optim.step()
    hm.record_step()


def test_collect_returns_expected_keys_after_one_step() -> None:
    model = TinyMLP()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    hm = HookManager()
    hm.attach(model)
    _fwd_bwd_step(model, optim, hm)
    metrics = hm.collect()
    d = metrics.to_dict()
    for key in (
        "per_layer_grad_norm",
        "max_layer_grad_norm",
        "min_layer_grad_norm",
        "dead_relu_fraction",
        "update_to_param_ratio",
        "grad_noise_scale",
        "step_count",
    ):
        assert key in d, f"missing {key}"
    assert metrics.step_count == 1


def test_dead_relu_fraction_matches_manual_count() -> None:
    torch.manual_seed(0)
    # A layer with a negative bias guarantees zeros after ReLU
    layer = nn.Linear(4, 6)
    with torch.no_grad():
        layer.bias.fill_(-10.0)
        layer.weight.fill_(0.0)
    model = nn.Sequential(layer, nn.ReLU())
    hm = HookManager()
    hm.attach(model)
    x = torch.randn(2, 4)
    y = model(x)
    _ = y.sum()
    metrics = hm.collect()
    assert metrics.dead_relu_fraction == pytest.approx(1.0)


def test_gradient_norm_matches_manual_value() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    hm = HookManager()
    hm.attach(model)
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    optim.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    manual_norms = {
        n: float(p.grad.norm().item()) for n, p in model.named_parameters()
    }
    optim.step()
    hm.record_step()
    metrics = hm.collect()
    for n, expected in manual_norms.items():
        assert metrics.per_layer_grad_norm[n] == pytest.approx(expected, rel=1e-6)


def test_detach_removes_hooks_cleanly() -> None:
    model = TinyMLP()
    hm = HookManager()
    hm.attach(model)
    assert len(hm._handles) > 0
    hm.detach()
    assert len(hm._handles) == 0
    # re-attaching a fresh manager must work without interference
    hm2 = HookManager()
    hm2.attach(model)
    assert len(hm2._handles) > 0
    hm2.detach()


def test_update_to_param_ratio_nonzero_after_step() -> None:
    torch.manual_seed(0)
    model = TinyMLP()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    hm = HookManager()
    hm.attach(model)
    _fwd_bwd_step(model, optim, hm)
    metrics = hm.collect()
    assert metrics.update_to_param_ratio > 0.0


def test_double_attach_raises() -> None:
    model = TinyMLP()
    hm = HookManager()
    hm.attach(model)
    with pytest.raises(RuntimeError):
        hm.attach(model)
    hm.detach()
