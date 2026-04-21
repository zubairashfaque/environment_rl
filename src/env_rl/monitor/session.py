"""Monitor session lifecycle.

This module owns the global session state. The public API below is the ONLY
legitimate logging path: the LLM has no other way to reach ``/judge_logs/``.
Every call validates session state and surfaces exceptions — a broken log
chain is unrecoverable, so silent continuation is worse than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from env_rl.monitor import rules as _rules
from env_rl.monitor.hooks import HookManager
from env_rl.monitor.logging import ChainedJsonlWriter


class SessionError(Exception):
    """Raised on lifecycle violations (duplicate start, attach without session, etc.)."""


ALLOWED_EVENT_TYPES = (
    "hyperparameter_change",
    "architecture_change",
    "rule_triggered_no_action",
)

METRICS_LOG = "metrics_log.jsonl"
DECISION_LOG = "decision_log.jsonl"
RULE_EVAL_LOG = "rule_evaluations.jsonl"


@dataclass
class _Session:
    log_dir: Path
    monitor_config: Mapping[str, Any]
    run_config: Mapping[str, Any]
    hook_manager: HookManager
    metrics_writer: ChainedJsonlWriter
    decisions_writer: ChainedJsonlWriter
    rule_evals_writer: ChainedJsonlWriter
    history: list[dict[str, Any]] = field(default_factory=list)
    rule_fire_log: list[dict[str, bool]] = field(default_factory=list)
    current_epoch: int = -1


_SESSION: _Session | None = None


def _active() -> _Session:
    if _SESSION is None:
        raise SessionError("no active session; call start_session() first")
    return _SESSION


def start_session(
    run_config: Mapping[str, Any],
    *,
    monitor_config: Mapping[str, Any],
) -> None:
    """Open the three append-only chained logs and record session_start."""
    global _SESSION
    if _SESSION is not None:
        raise SessionError("session already active; call end_session() first")

    log_dir = Path(monitor_config.get("log_dir", "./judge_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    root_hash = str(monitor_config["root_hash"])

    metrics_writer = ChainedJsonlWriter(log_dir / METRICS_LOG, root_hash=root_hash)
    decisions_writer = ChainedJsonlWriter(log_dir / DECISION_LOG, root_hash=root_hash)
    rule_evals_writer = ChainedJsonlWriter(log_dir / RULE_EVAL_LOG, root_hash=root_hash)

    start_rec = {"kind": "session_start", "run_config": dict(run_config)}
    metrics_writer.append(start_rec)
    decisions_writer.append(start_rec)
    rule_evals_writer.append(start_rec)

    _SESSION = _Session(
        log_dir=log_dir,
        monitor_config=dict(monitor_config),
        run_config=dict(run_config),
        hook_manager=HookManager(),
        metrics_writer=metrics_writer,
        decisions_writer=decisions_writer,
        rule_evals_writer=rule_evals_writer,
    )


def attach(model: Any) -> None:
    """Register diagnostic hooks on the model. Must come immediately after
    model init and before the first forward pass."""
    _active().hook_manager.attach(model)


def record_step() -> None:
    """Call after optimizer.step() in each training batch; captures the
    per-step gradient snapshot used for R2/R6/R7 and the update-to-param ratio."""
    _active().hook_manager.record_step()


def collect_epoch_metrics(
    model: Any,
    val_loader: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Aggregate the epoch's diagnostic snapshot from hook state.

    Extra kwargs (``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``,
    ``lr``, ``batch_size`` …) are merged in verbatim so the agent can record
    any training-loop state it tracks. The epoch number is stamped by the
    monitor — not supplied by the agent.
    """
    session = _active()
    hook_snapshot = session.hook_manager.collect().to_dict()
    session.current_epoch += 1
    metrics: dict[str, Any] = {
        "epoch": session.current_epoch,
        **hook_snapshot,
    }
    metrics.update(extra)
    if val_loader is not None:
        metrics.update(_evaluate_validation(model, val_loader))
    return metrics


def evaluate_rules(metrics: Mapping[str, Any]) -> dict[str, bool]:
    """Canonical rule evaluation for the latest epoch.

    The LLM gets to read the result; it never writes it. The metrics dict is
    appended to the monitor's internal history so EMA + 3-epoch-persistence
    calculations include every past epoch.
    """
    session = _active()
    session.history.append(dict(metrics))
    return _rules.evaluate_rules(session.history, session.monitor_config)


def log_epoch(metrics: Mapping[str, Any]) -> None:
    session = _active()
    session.metrics_writer.append({"kind": "epoch", **dict(metrics)})


def log_rule_eval(evals: Mapping[str, bool]) -> None:
    session = _active()
    evals_d = {k: bool(v) for k, v in evals.items()}
    session.rule_evals_writer.append({"kind": "rule_eval", "evals": evals_d})
    session.rule_fire_log.append(evals_d)


def log_decision(
    event_type: str,
    cites: Iterable[str],
    justification: str,
    **extra: Any,
) -> None:
    """Log an action (or deliberate non-action) responding to fired rules.

    Every cited rule must have fired in the ±2-epoch window ending at the
    current epoch. Citations that don't match the log raise ValueError — the
    monitor refuses to record fabricated attribution.
    """
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {ALLOWED_EVENT_TYPES}, got {event_type!r}"
        )

    session = _active()
    cites = list(cites)
    current = session.current_epoch
    window_start = max(0, current - 2)
    for rule in cites:
        fired = any(
            (e < len(session.rule_fire_log))
            and bool(session.rule_fire_log[e].get(rule, False))
            for e in range(window_start, current + 1)
        )
        if not fired:
            raise ValueError(
                f"rule {rule} was not fired in epochs "
                f"[{window_start}, {current}]; cannot cite"
            )

    session.decisions_writer.append(
        {
            "kind": "decision",
            "epoch": current,
            "event_type": event_type,
            "cites": cites,
            "justification": str(justification),
            **dict(extra),
        }
    )


def end_session() -> None:
    """Write bookend, close all writers, detach hooks, clear session state."""
    global _SESSION
    session = _active()
    end_rec = {"kind": "session_end"}
    session.metrics_writer.append(end_rec)
    session.decisions_writer.append(end_rec)
    session.rule_evals_writer.append(end_rec)
    session.metrics_writer.close()
    session.decisions_writer.close()
    session.rule_evals_writer.close()
    session.hook_manager.detach()
    _SESSION = None


def _reset_for_tests() -> None:
    """Test-only helper: clear the module-level session. Never call in production."""
    global _SESSION
    if _SESSION is not None:
        try:
            end_session()
        except Exception:  # noqa: BLE001
            _SESSION = None


def _evaluate_validation(model: Any, val_loader: Any) -> dict[str, float]:
    """Run validation pass over ``val_loader`` and return ``{val_loss, val_acc}``.

    Kept lightweight and framework-agnostic at the import level so unit tests
    that don't exercise this path don't need torch available.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_count += int(y.numel())
    model.train()
    if total_count == 0:
        return {"val_loss": 0.0, "val_acc": 0.0}
    return {
        "val_loss": total_loss / total_count,
        "val_acc": total_correct / total_count,
    }
