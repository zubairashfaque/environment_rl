"""Reference training run that exercises the playbook.

This is the ground-truth "a disciplined LLM could do this" run. It is NOT a
replacement for a real agent — it's a deterministic demonstrator that the
monitor and judge both function correctly end-to-end.

It drives a small CNN with deliberately aggressive initial hyperparameters
(lr=0.3, momentum-free SGD) so R7 (exploding gradients) fires within the
first few epochs, then applies the playbook remedy and continues training
with a gentler schedule. For rule IDs that do not naturally fire in 10–20
epochs on CIFAR-10, no decision is forced — the judge counts only actual
firings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_rl import monitor
from env_rl.agent.model import ResidualCNN


@dataclass
class ReferenceRunConfig:
    seed: int = 42
    lr: float = 0.3
    batch_size: int = 128
    momentum: float = 0.9
    weight_decay: float = 5e-4
    max_epochs: int = 10
    warmup_epochs: int = 0
    num_blocks: int = 6
    base_channels: int = 32
    activation: str = "relu"
    bn_enabled: bool = True
    workspace: str = "./workspace"


def _make_synthetic_loader(
    n_batches: int, batch_size: int, *, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Tiny synthetic CIFAR-like dataset for smoke tests (no download)."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (batch_size,), generator=g)
        batches.append((x, y))
    return batches


def _highest_precedence(fired: dict[str, bool]) -> str | None:
    """Pick the rule the playbook says we must action first."""
    from env_rl.judge.coverage import PRECEDENCE
    candidates = [r for r, f in fired.items() if f]
    if not candidates:
        return None
    return max(candidates, key=lambda r: PRECEDENCE.get(r, 0))


def _remedy_for(rule: str, current_lr: float) -> tuple[str, dict[str, Any], float]:
    """Prescribe a remedy per the playbook.

    Returns ``(event_type, extra_metadata, new_lr)``.
    """
    if rule == "R7":
        return (
            "hyperparameter_change",
            {"remedy_direction": "decrease_lr", "lr_new": current_lr / 10},
            current_lr / 10,
        )
    if rule == "R6":
        return (
            "hyperparameter_change",
            {"remedy_direction": "add_bn_or_residual", "note": "warmup LR"},
            current_lr,
        )
    if rule == "R1":
        return (
            "hyperparameter_change",
            {"remedy_direction": "decrease_lr", "lr_new": current_lr / 3},
            current_lr / 3,
        )
    if rule == "R2":
        return (
            "hyperparameter_change",
            {"remedy_direction": "increase_batch_size"},
            current_lr,
        )
    if rule == "R3":
        return (
            "hyperparameter_change",
            {"remedy_direction": "stop"},
            current_lr,
        )
    if rule == "R5":
        return (
            "architecture_change",
            {"remedy_direction": "swap_activation",
             "edit": {"op": "swap_activation", "to": "leaky_relu"}},
            current_lr,
        )
    if rule == "R4":
        return (
            "architecture_change",
            {"remedy_direction": "add_capacity",
             "edit": {"op": "add_block"}},
            current_lr,
        )
    return ("rule_triggered_no_action", {}, current_lr)


def run_reference(
    cfg: ReferenceRunConfig,
    train_loader: Iterable,
    val_loader: Iterable,
    monitor_config: dict,
    *,
    policy: Any = None,
    training_trace_path: Any = None,
) -> dict[str, Any]:
    """Run the reference training loop. Returns final summary dict.

    ``policy`` is optional. When provided, it must implement the
    ``DecisionPolicy`` protocol (see :mod:`env_rl.harness.policy`). The loop
    asks ``policy.decide(...)`` for each epoch where any rule fires, and
    applies the returned remedy (LR change / architecture edit) via the
    standard monitor API. When ``policy`` is None, the original hard-coded
    heuristic is used.

    Writes deliverables to ``cfg.workspace``; assumes the monitor's log_dir
    has been configured upstream.
    """
    torch.manual_seed(cfg.seed)
    workspace = Path(cfg.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Human-readable training trace (per-epoch summary + decisions + remedies)
    _trace_path = Path(training_trace_path) if training_trace_path else None
    if _trace_path is not None:
        _trace_path.parent.mkdir(parents=True, exist_ok=True)
        _trace_path.write_text("")  # truncate

    def _trace(record: dict[str, Any]) -> None:
        if _trace_path is None:
            return
        with open(_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    model = ResidualCNN(
        num_blocks=cfg.num_blocks,
        base_channels=cfg.base_channels,
        activation=cfg.activation,
        bn_enabled=cfg.bn_enabled,
    )
    monitor.start_session(
        {
            "seed": cfg.seed,
            "initial_hparams": {
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "momentum": cfg.momentum,
                "weight_decay": cfg.weight_decay,
            },
            "max_epochs": cfg.max_epochs,
        },
        monitor_config=monitor_config,
    )
    monitor.attach(model)
    _trace({"kind": "session_start", "cfg": {
        "seed": cfg.seed, "lr": cfg.lr, "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs, "num_blocks": cfg.num_blocks,
        "base_channels": cfg.base_channels, "activation": cfg.activation,
    }})

    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    current_lr = cfg.lr
    current_batch_size = cfg.batch_size
    current_activation = cfg.activation
    best_val_acc = 0.0
    best_state = None
    train_loader_list = list(train_loader)
    metrics_history: list[dict[str, Any]] = []

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        for x, y in train_loader_list:
            optim.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                # R7 will fire from metrics; keep going so it's captured.
                epoch_loss_sum += float("nan")
                break
            loss.backward()
            optim.step()
            monitor.record_step()
            epoch_loss_sum += float(loss.item()) * y.numel()
            epoch_correct += int((logits.argmax(-1) == y).sum().item())
            epoch_total += int(y.numel())

        train_loss = epoch_loss_sum / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)

        metrics = monitor.collect_epoch_metrics(
            model,
            val_loader=val_loader,
            train_loss=train_loss,
            train_acc=train_acc,
            lr=current_lr,
            batch_size=current_batch_size,
        )
        rule_evals = monitor.evaluate_rules(metrics)
        monitor.log_epoch(metrics)
        monitor.log_rule_eval(rule_evals)
        metrics_history.append(dict(metrics))

        _trace({
            "kind": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": metrics.get("val_loss"),
            "val_acc": metrics.get("val_acc"),
            "lr": current_lr,
            "max_grad": metrics.get("max_layer_grad_norm"),
            "min_grad": metrics.get("min_layer_grad_norm"),
            "dead_relu": metrics.get("dead_relu_fraction"),
            "gns": metrics.get("grad_noise_scale"),
            "fired_rules": sorted([r for r, v in rule_evals.items() if v]),
        })

        # Respond to fired rules — honor precedence by actioning the top class
        # and deferring everything else explicitly.
        top = _highest_precedence(rule_evals)
        if top is not None:
            if policy is not None:
                decision = policy.decide(
                    top_rule=top,
                    all_fired=dict(rule_evals),
                    metrics=metrics,
                    epoch=epoch,
                    current_lr=current_lr,
                    current_batch_size=current_batch_size,
                    recent_history=metrics_history,
                )
                # If the LLM proposed an architecture edit, try to apply it.
                # Unsupported or no-op edits are downgraded to
                # rule_triggered_no_action so the log never claims an edit
                # happened when none did (judge step 6 invariant).
                if decision.event_type == "architecture_change":
                    from env_rl.harness.edits import apply_edit_in_place, is_supported
                    edit = decision.remedy_params.get("edit") or {}
                    op = edit.get("op", "none") if edit else "none"
                    if op == "none" or not is_supported(edit):
                        decision.event_type = "rule_triggered_no_action"
                        decision.justification = (
                            f"harness does not execute edit {op!r}; "
                            f"deferring {top} (run longer or action an LR remedy)"
                        )
                        decision.remedy_params = {}
                    else:
                        try:
                            apply_edit_in_place(model, edit)
                            if op == "swap_activation":
                                current_activation = str(edit.get("to", current_activation)).lower()
                        except ValueError:
                            decision.event_type = "rule_triggered_no_action"
                            decision.justification = (
                                f"edit {op!r} failed to apply for {top}; deferring"
                            )
                            decision.remedy_params = {}
                monitor.log_decision(**decision.to_log_kwargs())
                _trace({
                    "kind": "decision",
                    "epoch": epoch,
                    "source": "policy",
                    "cited_rule": top,
                    "event_type": decision.event_type,
                    "justification": decision.justification[:200],
                    "remedy_direction": decision.remedy_direction,
                    "remedy_params": decision.remedy_params,
                })
                # Apply remedy if the policy specified a new LR
                lr_new = decision.remedy_params.get("lr_new")
                if lr_new is not None and lr_new != current_lr and lr_new > 0:
                    for g in optim.param_groups:
                        g["lr"] = float(lr_new)
                    current_lr = float(lr_new)
                    _trace({"kind": "remedy_applied", "epoch": epoch,
                            "change": "lr", "from": current_lr / (lr_new / current_lr) if lr_new else 0,
                            "to": lr_new})
            else:
                event_type, extra, new_lr = _remedy_for(top, current_lr)
                monitor.log_decision(
                    event_type,
                    cites=[top],
                    justification=f"auto-remedy for {top} at epoch {epoch}",
                    **extra,
                )
                if new_lr != current_lr:
                    for g in optim.param_groups:
                        g["lr"] = new_lr
                    current_lr = new_lr
            # Defer every other fired rule.
            for rule, fired in rule_evals.items():
                if fired and rule != top:
                    monitor.log_decision(
                        "rule_triggered_no_action",
                        cites=[rule],
                        justification=f"deferred_to_{top}",
                    )

        if metrics.get("val_acc", 0.0) > best_val_acc:
            best_val_acc = float(metrics["val_acc"])
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    monitor.end_session()
    _trace({"kind": "session_end", "best_val_acc": best_val_acc})

    # write deliverables
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, workspace / "best_model.pt")

    (workspace / "model.py").write_text(_model_py_source(cfg, activation=current_activation))
    (workspace / "run_config.json").write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "initial_hparams": {
                    "lr": cfg.lr,
                    "batch_size": cfg.batch_size,
                    "momentum": cfg.momentum,
                    "weight_decay": cfg.weight_decay,
                },
                "max_epochs": cfg.max_epochs,
            },
            indent=2,
        )
    )

    return {
        "best_val_acc": best_val_acc,
        "final_epoch": cfg.max_epochs - 1,
    }


def _model_py_source(cfg: ReferenceRunConfig, *, activation: str | None = None) -> str:
    # Always use the FINAL activation (after any swaps) so load_model() returns
    # a model whose spec() matches the judge's architecture replay.
    act = activation if activation is not None else cfg.activation
    return (
        "import torch\n"
        "from env_rl.agent.model import ResidualCNN\n"
        "\n"
        "def load_model():\n"
        f"    m = ResidualCNN(num_blocks={cfg.num_blocks}, "
        f"base_channels={cfg.base_channels}, "
        f"activation={act!r}, "
        f"bn_enabled={cfg.bn_enabled})\n"
        "    state = torch.load(str(__file__).rsplit('/', 1)[0] + '/best_model.pt')\n"
        "    m.load_state_dict(state)\n"
        "    m.eval()\n"
        "    return m\n"
    )
