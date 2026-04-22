"""Apply structural edits from a Decision to a live ``nn.Module``.

Only edits that the harness can safely execute mid-training are supported.
Unsupported edits are reported so the caller can downgrade the Decision to
``rule_triggered_no_action`` — which keeps the log and the model consistent
(the invariant enforced by judge step 6).

Currently supported:
  - ``swap_activation`` with ``to`` in {"relu", "leaky_relu", "gelu"}

Intentionally NOT supported (would require optimizer rebuild or param
reshape, which makes mid-run adjustment fragile):
  - ``add_block`` / ``remove_block``
  - ``add_bn`` / ``remove_bn``
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

SUPPORTED_EDITS = frozenset({"swap_activation", "none"})
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


def apply_edit_in_place(model: nn.Module, edit: dict[str, Any]) -> None:
    """Mutate the model to reflect ``edit``. Raises ``ValueError`` if unsupported.

    The caller is responsible for making sure supported() returns True first
    and for downgrading the Decision otherwise.
    """
    op = edit.get("op", "none")
    if op == "none":
        return
    if op != "swap_activation":
        raise ValueError(f"harness does not execute edit op {op!r}")

    target = str(edit.get("to", "")).lower()
    if target not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported activation {target!r}")

    new_cls = _ACTIVATION_CLS[target]

    # Traverse and replace every ReLU-like activation with the new class.
    # Also track what the model.spec() should report.
    _RELU_LIKE = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.PReLU)
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, _RELU_LIKE):
                setattr(parent, name, new_cls())

    # Update the model's own spec markers so judge step 6 sees the new state.
    if hasattr(model, "_activation"):
        setattr(model, "_activation", target)
