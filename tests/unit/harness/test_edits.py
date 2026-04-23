"""Tests for the structural edit applier (swap_activation, add_block, remove_block)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.agent.model import ResidualCNN  # noqa: E402
from env_rl.harness.edits import (  # noqa: E402
    CONTINUE_EDITS,
    RESTART_EDITS,
    SUPPORTED_EDITS,
    apply_edit_in_place,
    is_restart_edit,
    is_supported,
)


def _make_model():
    return ResidualCNN(
        num_blocks=2, base_channels=8, activation="relu", bn_enabled=True
    )


# -- is_supported / schema coverage ------------------------------------------


def test_supported_edits_include_add_and_remove_block() -> None:
    assert "add_block" in SUPPORTED_EDITS
    assert "remove_block" in SUPPORTED_EDITS
    assert "swap_activation" in SUPPORTED_EDITS


def test_is_supported_swap_activation_to_leaky_relu() -> None:
    assert is_supported({"op": "swap_activation", "to": "leaky_relu"})


def test_is_supported_rejects_unknown_op() -> None:
    assert not is_supported({"op": "turn_inside_out"})


# -- swap_activation ---------------------------------------------------------


def test_swap_activation_mutates_spec_and_modules() -> None:
    model = _make_model()
    assert model.spec()["activation"] == "relu"
    apply_edit_in_place(model, {"op": "swap_activation", "to": "leaky_relu"})
    assert model.spec()["activation"] == "leaky_relu"
    # no ReLU left anywhere
    assert not any(isinstance(m, nn.ReLU) for m in model.modules())


# -- add_block ---------------------------------------------------------------


def test_add_block_appends_and_updates_spec() -> None:
    model = _make_model()
    assert model.spec()["num_blocks"] == 2
    apply_edit_in_place(model, {"op": "add_block"})
    assert model.spec()["num_blocks"] == 3
    assert len(model.blocks) == 3


def test_add_block_registers_new_params_with_optimizer() -> None:
    model = _make_model()
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    initial_groups = len(optim.param_groups)

    apply_edit_in_place(model, {"op": "add_block"}, optimizer=optim)

    # The new block's params should live in a fresh param group with the
    # same LR as the base group.
    assert len(optim.param_groups) == initial_groups + 1
    assert optim.param_groups[-1]["lr"] == pytest.approx(0.1)


def test_add_block_new_block_has_gradients_after_backward() -> None:
    model = _make_model()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    apply_edit_in_place(model, {"op": "add_block"}, optimizer=optim)
    new_block = model.blocks[-1]

    x = torch.randn(2, 8, 8, 8)  # matches stem output shape
    y = model.blocks[0](x)  # warm through old blocks manually not needed
    # actually run through the whole model for a clean gradient
    img = torch.randn(2, 3, 32, 32)
    logits = model(img)
    loss = logits.sum()
    loss.backward()
    for p in new_block.parameters():
        assert p.grad is not None


def test_add_block_forward_shape_stable() -> None:
    model = _make_model()
    apply_edit_in_place(model, {"op": "add_block"})
    out = model(torch.randn(3, 3, 32, 32))
    assert out.shape == (3, 10)


# -- remove_block ------------------------------------------------------------


def test_remove_block_shrinks_and_updates_spec() -> None:
    model = _make_model()
    apply_edit_in_place(model, {"op": "remove_block"})
    assert model.spec()["num_blocks"] == 1
    assert len(model.blocks) == 1


def test_remove_block_refuses_to_empty_the_model() -> None:
    model = ResidualCNN(num_blocks=1, base_channels=8)
    with pytest.raises(ValueError):
        apply_edit_in_place(model, {"op": "remove_block"})


def test_remove_block_forward_still_works() -> None:
    model = _make_model()
    apply_edit_in_place(model, {"op": "remove_block"})
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


# -- RESTART vs CONTINUE classification --------------------------------------


def test_continue_edits_set_contains_swap_and_none() -> None:
    assert "swap_activation" in CONTINUE_EDITS
    assert "none" in CONTINUE_EDITS


def test_restart_edits_set_contains_block_and_bn_changes() -> None:
    assert "add_block" in RESTART_EDITS
    assert "remove_block" in RESTART_EDITS
    assert "add_bn" in RESTART_EDITS


def test_restart_and_continue_sets_are_disjoint() -> None:
    assert CONTINUE_EDITS.isdisjoint(RESTART_EDITS)


def test_is_restart_edit_swap_activation_false() -> None:
    assert not is_restart_edit({"op": "swap_activation", "to": "leaky_relu"})


def test_is_restart_edit_none_false() -> None:
    assert not is_restart_edit({"op": "none"})
    assert not is_restart_edit({})  # default op is "none"


def test_is_restart_edit_add_block_true() -> None:
    assert is_restart_edit({"op": "add_block"})


def test_is_restart_edit_remove_block_true() -> None:
    assert is_restart_edit({"op": "remove_block"})


def test_is_restart_edit_add_bn_true() -> None:
    assert is_restart_edit({"op": "add_bn"})
