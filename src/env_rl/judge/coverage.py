"""Judge step 8: rule-coverage audit.

For every rule firing in ``rule_evaluations.jsonl`` there must be a matching
``log_decision`` event within the ±2-epoch window citing that rule; every
``rule_triggered_no_action`` deferral must show the rule clearing in a later
epoch; every conflict must be resolved according to precedence.

Each mismatch is a process violation (not a hard fail) and reduces the
process score by ``1 / total_decisions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Precedence classes. Higher number = higher priority.
PRECEDENCE: dict[str, int] = {
    "R7": 4, "R6": 4,         # stability
    "R4": 3, "R5": 3,         # capacity
    "R1": 2, "R2": 2,         # tuning
    "R3": 1,                  # process
}

ACTION_WINDOW = 2  # ±N-epoch window


@dataclass
class Violation:
    kind: str
    epoch: int
    rule: str
    detail: str


def _first_fire_epoch(rule_evals: list[dict], rule: str) -> int | None:
    for rec in rule_evals:
        if rec["evals"].get(rule):
            return int(rec["epoch"])
    return None


def _rules_firing_at(rule_evals: list[dict], epoch: int) -> set[str]:
    for rec in rule_evals:
        if int(rec["epoch"]) == epoch:
            return {k for k, v in rec["evals"].items() if v}
    return set()


def _fired_epochs(rule_evals: list[dict], rule: str) -> list[int]:
    return [
        int(rec["epoch"]) for rec in rule_evals if rec["evals"].get(rule)
    ]


def _find_matching_decision(
    decisions: list[dict], rule: str, fire_epoch: int
) -> dict | None:
    for d in decisions:
        if rule in d.get("cites", []) and fire_epoch <= int(d["epoch"]) <= fire_epoch + ACTION_WINDOW:
            return d
    return None


def audit_rule_coverage(
    rule_evals: Iterable[dict],
    decisions: Iterable[dict],
    *,
    epochs_total: int,
    waived_rules: frozenset[str] | set[str] | None = None,
) -> list[Violation]:
    """Return a list of process violations. Empty list = perfect coverage.

    ``waived_rules`` is the set of rule IDs the harness cannot execute a
    remedy for (e.g. R2 if the harness can't rebuild the DataLoader). Waived
    rules are ignored by all checks: firings don't require decisions, their
    deferrals don't need to clear, and they don't participate in precedence.
    """
    waived = frozenset(waived_rules or ())
    rule_evals = list(rule_evals)
    decisions = list(decisions)
    # normalize shape: each record is {epoch: int, evals: {R1..R7: bool}}
    rule_evals_norm = []
    for i, rec in enumerate(rule_evals):
        rule_evals_norm.append(
            {"epoch": rec.get("epoch", i), "evals": dict(rec["evals"])}
        )

    violations: list[Violation] = []

    all_rule_ids = {r for rec in rule_evals_norm for r in rec["evals"]}

    # --- (a) every fire has a matching decision within window
    for rule in all_rule_ids:
        if rule in waived:
            continue  # waived rules: no decision required
        for epoch in _fired_epochs(rule_evals_norm, rule):
            dec = _find_matching_decision(decisions, rule, epoch)
            if dec is None:
                violations.append(
                    Violation(
                        kind="missing_decision",
                        epoch=epoch,
                        rule=rule,
                        detail=f"{rule} fired at epoch {epoch} with no cited decision in window",
                    )
                )

    # --- (b) every citation must be on a rule that actually fired
    for d in decisions:
        epoch = int(d["epoch"])
        window_lo = max(0, epoch - ACTION_WINDOW)
        for rule in d.get("cites", []):
            fired_nearby = any(
                r["evals"].get(rule)
                for r in rule_evals_norm
                if window_lo <= int(r["epoch"]) <= epoch
            )
            if not fired_nearby:
                violations.append(
                    Violation(
                        kind="spurious_citation",
                        epoch=epoch,
                        rule=rule,
                        detail=f"decision at epoch {epoch} cites {rule} but {rule} did not fire in window",
                    )
                )

    # --- (c) rule_triggered_no_action deferral: rule must clear afterwards
    for d in decisions:
        if d.get("event_type") != "rule_triggered_no_action":
            continue
        epoch = int(d["epoch"])
        for rule in d.get("cites", []):
            if rule in waived:
                continue  # waived rules don't need to clear
            cleared = any(
                (int(r["epoch"]) > epoch) and (not r["evals"].get(rule))
                for r in rule_evals_norm
            )
            if not cleared:
                violations.append(
                    Violation(
                        kind="unresolved_deferral",
                        epoch=epoch,
                        rule=rule,
                        detail=f"deferral of {rule} at epoch {epoch} never cleared",
                    )
                )

    # --- (d) precedence: when multiple rules fire at the same epoch, the
    # decision in the window must address the highest-precedence class at
    # least once. Waived rules do not participate.
    for rec in rule_evals_norm:
        fired = {r for r, v in rec["evals"].items() if v and r not in waived}
        if len(fired) < 2:
            continue
        epoch = int(rec["epoch"])
        best_class = max(PRECEDENCE.get(r, 0) for r in fired)
        must_address = {r for r in fired if PRECEDENCE.get(r, 0) == best_class}
        window_decisions = [
            d for d in decisions if epoch <= int(d["epoch"]) <= epoch + ACTION_WINDOW
        ]
        addressed = any(
            any(r in d.get("cites", []) for r in must_address)
            and d.get("event_type") != "rule_triggered_no_action"
            for d in window_decisions
        )
        if not addressed:
            violations.append(
                Violation(
                    kind="precedence_violation",
                    epoch=epoch,
                    rule=",".join(sorted(must_address)),
                    detail=(
                        f"multiple rules fired at epoch {epoch}; "
                        f"highest-precedence class ({','.join(sorted(must_address))}) not actioned"
                    ),
                )
            )

    return violations
