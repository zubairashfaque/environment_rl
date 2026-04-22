"""Apply structural edits from a Decision to a live ``nn.Module``.

Only edits that the harness can safely execute mid-training are supported.
Unsupported edits are reported so the caller can downgrade the Decision to
``rule_triggered_no_action`` — which keeps the log and the model consistent
(the invariant enforced by judge step 6).

Currently supported:
  - ``swap_activation`` with ``to`` in {"relu", "leaky_relu", "gelu"}
  - ``add_block`` — append a fresh ResBlock to ``model.blocks`` and register
    its parameters with the optimizer (if provided)
  - ``remove_block`` — pop the last block from ``model.blocks``; the
    optimizer keeps the (now unused) param group harmlessly

Intentionally NOT supported yet (require BN/residual retrofit that changes
tensor shapes):
  - ``add_bn`` / ``remove_bn``
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

SUPPORTED_EDITS = frozenset({"swap_activation", "add_block", "remove_block", "none"})
SUPPORTED_ACTIVATIONS = frozenset({"relu", "leaky_relu", "gelu"})

_ACTIVATION_CLS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),  # type: ignore[dict-item]
    "gelu": nn.GELU,
}


def is_supported(edit: dict[str, Any]) -> bool:
    op = edit.get("op", "none")
    if op not in SUPPORTED_EDITS:
        return False
    if op == "swap_activation":
        target = str(edit.get("to", "")).lower()
        return target in SUPPORTED_ACTIVATIONS
    return True


def apply_edit_in_place(
    model: nn.Module,
    edit: dict[str, Any],
    optimizer: Any = None,
) -> None:
    """Mutate the model to reflect ``edit``. Raises ``ValueError`` if unsupported.

    If ``optimizer`` is supplied, newly-added parameters (e.g. from
    ``add_block``) are registered as a fresh param group using the current
    first param group's settings (typically the training LR).
    """
    op = edit.get("op", "none")
    if op == "none":
        return

    if op == "swap_activation":
        _apply_swap_activation(model, edit)
        return
    if op == "add_block":
        _apply_add_block(model, optimizer)
        return
    if op == "remove_block":
        _apply_remove_block(model)
        return

    raise ValueError(f"harness does not execute edit op {op!r}")


def _apply_swap_activation(model: nn.Module, edit: dict[str, Any]) -> None:
    target = str(edit.get("to", "")).lower()
    if target not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported activation {target!r}")
    new_cls = _ACTIVATION_CLS[target]

    _RELU_LIKE = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.PReLU)
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, _RELU_LIKE):
                setattr(parent, name, new_cls())

    if hasattr(model, "_activation"):
        setattr(model, "_activation", target)


def _apply_add_block(model: nn.Module, optimizer: Any) -> None:
    """Append one more ResBlock with the same channels/activation/BN as existing."""
    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise ValueError("model must expose `blocks: nn.ModuleList` to add_block")
    if len(model.blocks) == 0:
        raise ValueError("cannot add_block: no existing template block")

    from env_rl.agent.model import ResBlock, he_init_
    template = model.blocks[-1]
    # Infer channels from the template's first conv
    channels = int(template.conv1.in_channels)
    activation = getattr(model, "_activation", "relu")
    bn_enabled = bool(getattr(model, "_bn_enabled", True))

    new_block = ResBlock(
        channels=channels,
        activation=activation,
        bn_enabled=bn_enabled,
    )
    he_init_(new_block)

    # Move to same device as rest of model
    first_param = next(model.parameters())
    new_block = new_block.to(device=first_param.device, dtype=first_param.dtype)

    model.blocks.append(new_block)
    if hasattr(model, "_num_blocks"):
        model._num_blocks = int(model._num_blocks) + 1

    if optimizer is not None:
        # Pick up the current LR from group 0 so the new block trains at the
        # same rate as the rest of the model.
        current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
        optimizer.add_param_group({
            "params": list(new_block.parameters()),
            "lr": current_lr,
        })


def _apply_remove_block(model: nn.Module) -> None:
    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise ValueError("model must expose `blocks: nn.ModuleList` to remove_block")
    if len(model.blocks) <= 1:
        raise ValueError("cannot remove_block: at least one block must remain")
    # Drop the last block; its parameters simply fall out of the forward pass.
    # The optimizer's param_group for those params becomes inert — harmless.
    model.blocks = nn.ModuleList(list(model.blocks)[:-1])
    if hasattr(model, "_num_blocks"):
        model._num_blocks = int(model._num_blocks) - 1
