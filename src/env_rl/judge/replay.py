"""Judge step 6: replay architecture_change events and match against the
submitted model.

The replay works on a compact, serializable spec (a dict) — not on the live
``nn.Module``. Each ``architecture_change`` decision contains an ``edit``
object that names an operation in the registry below; replaying applies the
ops in log order to the initial spec from ``run_config.json``.

The submitted model's spec is extracted structurally and compared to the
replay result. Mismatch = hard fail (``trained one model, logged another``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn

from env_rl.judge.deliverables import HardFail

# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

OpFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
_OPS: dict[str, OpFn] = {}


def register_op(name: str) -> Callable[[OpFn], OpFn]:
    def wrap(fn: OpFn) -> OpFn:
        _OPS[name] = fn
        return fn
    return wrap


@register_op("add_block")
def _add_block(spec: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    spec["num_blocks"] = int(spec.get("num_blocks", 0)) + 1
    return spec


@register_op("remove_block")
def _remove_block(spec: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    n = int(spec.get("num_blocks", 0))
    if n <= 0:
        raise HardFail("cannot remove_block: num_blocks already 0")
    spec["num_blocks"] = n - 1
    return spec


@register_op("swap_activation")
def _swap_activation(spec: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    target = str(edit["to"])
    spec["activation"] = target
    return spec


@register_op("add_bn")
def _add_bn(spec: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    spec["bn_enabled"] = True
    return spec


@register_op("remove_bn")
def _remove_bn(spec: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    spec["bn_enabled"] = False
    return spec


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_architecture_changes(
    initial_spec: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply each architecture_change edit to the initial spec in log order."""
    spec = copy.deepcopy(initial_spec)
    for ev in events:
        if ev.get("event_type") != "architecture_change":
            continue
        edit = ev.get("edit") or ev.get("payload", {}).get("edit")
        if edit is None:
            raise HardFail(
                f"architecture_change event missing 'edit': {ev!r}"
            )
        op_name = str(edit.get("op"))
        fn = _OPS.get(op_name)
        if fn is None:
            raise HardFail(f"unknown architecture edit op: {op_name!r}")
        spec = fn(spec, edit)
    return spec


# ---------------------------------------------------------------------------
# Structural extraction from a live model
# ---------------------------------------------------------------------------

_ACTIVATION_NAMES = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.GELU: "gelu",
    nn.PReLU: "prelu",
    nn.SiLU: "silu",
}


def extract_spec_from_model(model: nn.Module) -> dict[str, Any]:
    """Return the same dict shape produced by ``replay_architecture_changes``.

    The model must expose a ``spec()`` method returning a dict, OR standard
    attributes: ``num_blocks`` (int), ``activation_cls`` or an identifiable
    activation module, ``bn_enabled`` (bool). We prefer ``spec()`` because
    the reference agent can produce a faithful canonical form.
    """
    if hasattr(model, "spec") and callable(model.spec):
        return dict(model.spec())

    spec: dict[str, Any] = {}
    if hasattr(model, "num_blocks"):
        spec["num_blocks"] = int(getattr(model, "num_blocks"))

    # find first activation-like module
    for m in model.modules():
        name = _ACTIVATION_NAMES.get(type(m))
        if name is not None:
            spec["activation"] = name
            break

    # BN presence
    spec["bn_enabled"] = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    return spec


# ---------------------------------------------------------------------------
# High-level check
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    expected_spec: dict[str, Any]
    actual_spec: dict[str, Any]


def check_architecture_matches_submission(
    initial_spec: dict[str, Any],
    events: list[dict[str, Any]],
    submitted_model: nn.Module,
) -> ReplayResult:
    """Walk the replay; raise ``HardFail`` if it does not match the model."""
    expected = replay_architecture_changes(initial_spec, events)
    actual = extract_spec_from_model(submitted_model)

    # Compare only the keys the replay tracks — actual may carry extras.
    for k, v in expected.items():
        if actual.get(k) != v:
            raise HardFail(
                f"architecture mismatch on {k!r}: expected {v!r}, submitted {actual.get(k)!r}"
            )
    return ReplayResult(expected_spec=expected, actual_spec=actual)
