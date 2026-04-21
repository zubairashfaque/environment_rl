import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.judge.evaluation import evaluate_accuracy  # noqa: E402


def _tiny_loader(n: int = 16, n_classes: int = 2):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = torch.randint(0, n_classes, (n,))
    return [(x, y)]


class ConstNet(nn.Module):
    def __init__(self, target_class: int = 0, n_classes: int = 2) -> None:
        super().__init__()
        self.target = target_class
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], self.n_classes, device=x.device)
        logits[:, self.target] = 1.0
        return logits


def test_evaluate_accuracy_deterministic_on_fixed_model_and_batch() -> None:
    torch.manual_seed(0)
    loader = _tiny_loader(n=20, n_classes=2)
    model = ConstNet(target_class=0, n_classes=2)
    acc1 = evaluate_accuracy(model, loader)
    acc2 = evaluate_accuracy(model, loader)
    assert acc1 == acc2


def test_evaluate_accuracy_zero_on_empty_loader() -> None:
    model = ConstNet()
    assert evaluate_accuracy(model, []) == 0.0


def test_evaluate_accuracy_one_when_constnet_matches_labels() -> None:
    torch.manual_seed(0)
    loader = [(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))]
    model = ConstNet(target_class=0, n_classes=2)
    assert evaluate_accuracy(model, loader) == 1.0
