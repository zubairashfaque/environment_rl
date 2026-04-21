"""Judge steps 1-4: deliverables, signature, weight load, run_config consistency.

Each step in this module raises :class:`HardFail` on violation — any such
violation zeroes both the accuracy and process-integrity scores. The judge
never trusts anything the LLM writes in ``/workspace/`` beyond the three
deliverables listed here.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import torch.nn as nn

DELIVERABLES = ("model.py", "best_model.pt", "run_config.json")
REQUIRED_RUN_CONFIG_FIELDS = ("seed", "initial_hparams", "max_epochs")


class HardFail(Exception):
    """Fatal integrity violation. Zeroes both scores."""


def check_deliverables_exist(workspace: str | Path) -> None:
    """Step 1: all three files exist in /workspace/."""
    workspace = Path(workspace)
    missing = [name for name in DELIVERABLES if not (workspace / name).is_file()]
    if missing:
        raise HardFail(f"missing deliverables: {missing}")


def check_load_model_signature(workspace: str | Path) -> Any:
    """Step 2: model.py defines a zero-argument callable ``load_model``."""
    workspace = Path(workspace)
    model_py = workspace / "model.py"
    spec = importlib.util.spec_from_file_location("_submitted_model", model_py)
    if spec is None or spec.loader is None:
        raise HardFail(f"cannot import {model_py}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise HardFail(f"importing model.py raised: {type(e).__name__}: {e}") from e

    load_model = getattr(module, "load_model", None)
    if load_model is None or not callable(load_model):
        raise HardFail("model.py must define a callable `load_model`")

    sig = inspect.signature(load_model)
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if required:
        raise HardFail(
            f"load_model must take no required arguments; "
            f"found {[p.name for p in required]}"
        )
    return load_model


def load_submitted_model(workspace: str | Path) -> nn.Module:
    """Step 3: call ``load_model()`` and verify its return value."""
    load_model = check_load_model_signature(workspace)
    try:
        model = load_model()
    except Exception as e:
        raise HardFail(f"load_model() raised: {type(e).__name__}: {e}") from e

    if not isinstance(model, nn.Module):
        raise HardFail(
            f"load_model() must return nn.Module, got {type(model).__name__}"
        )
    if model.training:
        raise HardFail("load_model() must return a model in eval() mode")
    return model


def check_run_config(
    workspace: str | Path,
    session_start_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 4: ``run_config.json`` has the required fields and matches the
    ``session_start`` record that was logged at the top of the run."""
    workspace = Path(workspace)
    path = workspace / "run_config.json"
    try:
        cfg = json.loads(path.read_text())
    except Exception as e:
        raise HardFail(f"run_config.json unreadable: {e}") from e

    missing = [f for f in REQUIRED_RUN_CONFIG_FIELDS if f not in cfg]
    if missing:
        raise HardFail(f"run_config.json missing fields: {missing}")

    if session_start_record is not None:
        logged = session_start_record.get("run_config", {})
        for f in REQUIRED_RUN_CONFIG_FIELDS:
            if cfg.get(f) != logged.get(f):
                raise HardFail(
                    f"run_config.json field {f!r} ({cfg.get(f)!r}) does not match "
                    f"session_start logged value ({logged.get(f)!r})"
                )
    return cfg
