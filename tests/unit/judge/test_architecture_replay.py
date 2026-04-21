import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.judge.deliverables import HardFail  # noqa: E402
from env_rl.judge.replay import (  # noqa: E402
    check_architecture_matches_submission,
    extract_spec_from_model,
    replay_architecture_changes,
)


def _event(op: str, **kwargs) -> dict:
    return {"event_type": "architecture_change", "edit": {"op": op, **kwargs}}


class ToyNet(nn.Module):
    def __init__(self, num_blocks: int, activation: str, bn_enabled: bool) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self._bn_enabled = bn_enabled
        act_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation]
        layers: list[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(nn.Linear(4, 4))
            if bn_enabled:
                layers.append(nn.BatchNorm2d(4, track_running_stats=False))
            layers.append(act_cls())
        self.body = nn.Sequential(*layers)

    def spec(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "activation": "leaky_relu"
            if isinstance(self.body[-1], nn.LeakyReLU)
            else "relu",
            "bn_enabled": self._bn_enabled,
        }

    def forward(self, x):  # pragma: no cover
        return self.body(x)


def test_replay_of_three_edits_matches_expected_spec() -> None:
    initial = {"num_blocks": 4, "activation": "relu", "bn_enabled": False}
    events = [
        _event("add_block"),
        _event("swap_activation", to="leaky_relu"),
        _event("add_bn"),
    ]
    expected = replay_architecture_changes(initial, events)
    assert expected == {
        "num_blocks": 5,
        "activation": "leaky_relu",
        "bn_enabled": True,
    }


def test_replay_ignores_non_architecture_change_events() -> None:
    initial = {"num_blocks": 4, "activation": "relu", "bn_enabled": False}
    events = [
        {"event_type": "hyperparameter_change", "edit": {"op": "add_block"}},
        _event("add_block"),
    ]
    out = replay_architecture_changes(initial, events)
    assert out["num_blocks"] == 5


def test_replay_unknown_op_hard_fails() -> None:
    with pytest.raises(HardFail):
        replay_architecture_changes({"num_blocks": 1}, [_event("turn_inside_out")])


def test_replay_remove_beyond_zero_hard_fails() -> None:
    with pytest.raises(HardFail):
        replay_architecture_changes({"num_blocks": 0}, [_event("remove_block")])


def test_check_matches_submission_passes_on_agreement() -> None:
    initial = {"num_blocks": 4, "activation": "relu", "bn_enabled": False}
    events = [
        _event("add_block"),
        _event("swap_activation", to="leaky_relu"),
        _event("add_bn"),
    ]
    submitted = ToyNet(num_blocks=5, activation="leaky_relu", bn_enabled=True)
    result = check_architecture_matches_submission(initial, events, submitted)
    assert result.expected_spec["num_blocks"] == 5


def test_check_matches_submission_hard_fails_on_disagreement() -> None:
    initial = {"num_blocks": 4, "activation": "relu", "bn_enabled": False}
    events = [
        _event("add_block"),
        _event("swap_activation", to="leaky_relu"),
    ]
    # Replay says 5 blocks, leaky_relu; but submitted has 6 blocks
    submitted = ToyNet(num_blocks=6, activation="leaky_relu", bn_enabled=False)
    with pytest.raises(HardFail):
        check_architecture_matches_submission(initial, events, submitted)


def test_extract_spec_from_model_fallback_without_spec_method() -> None:
    class MiniNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_blocks = 2
            self.body = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        def forward(self, x):  # pragma: no cover
            return self.body(x)
    spec = extract_spec_from_model(MiniNet())
    assert spec["num_blocks"] == 2
    assert spec["activation"] == "relu"
    assert spec["bn_enabled"] is False
