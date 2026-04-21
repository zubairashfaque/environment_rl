"""Judge step 9: decision defensibility sampling.

Check sampled decisions for:
  - event_type vs rule-class consistency (e.g., R7 remedy is hyperparameter
    or architecture change, not something else)
  - an optional ``remedy_direction`` field must agree with the rule's
    prescribed remedy direction (e.g., R7 requires LR decrease, not increase)
  - pathological padding: trivially-tripped rules (a short single-epoch
    firing with a purely cosmetic action) are flagged

Each failure is a process violation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Rule → allowed event types (which actions are defensible for each)
ALLOWED_ACTIONS: dict[str, set[str]] = {
    "R1": {"hyperparameter_change", "rule_triggered_no_action"},
    "R2": {"hyperparameter_change", "rule_triggered_no_action"},
    "R3": {"hyperparameter_change", "rule_triggered_no_action"},
    "R4": {"architecture_change", "rule_triggered_no_action"},
    "R5": {"architecture_change", "rule_triggered_no_action"},
    "R6": {"hyperparameter_change", "architecture_change", "rule_triggered_no_action"},
    "R7": {"hyperparameter_change", "architecture_change", "rule_triggered_no_action"},
}

# Optional direction checks: the agent may include a ``remedy_direction``
# tag. If so, it must match.
EXPECTED_DIRECTION: dict[str, str] = {
    "R1_high": "decrease_lr",
    "R1_low": "increase_lr",
    "R2_low": "increase_batch_size",
    "R2_high": "decrease_batch_size",
    "R7": "decrease_lr",
    "R3": "stop",
    "R5": "swap_activation",
    "R6": "add_bn_or_residual",
    "R4": "add_capacity",
}


@dataclass
class DefensibilityFailure:
    kind: str
    epoch: int
    rule: str
    detail: str


def check_decision(decision: dict) -> list[DefensibilityFailure]:
    """Return any defensibility issues with this single decision."""
    issues: list[DefensibilityFailure] = []
    event_type = decision.get("event_type", "")
    epoch = int(decision.get("epoch", -1))
    cites = decision.get("cites", []) or []

    for rule in cites:
        allowed = ALLOWED_ACTIONS.get(rule, set())
        if event_type not in allowed:
            issues.append(
                DefensibilityFailure(
                    kind="event_type_mismatch",
                    epoch=epoch,
                    rule=rule,
                    detail=(
                        f"decision at epoch {epoch} cites {rule} with "
                        f"event_type={event_type!r}, expected one of {sorted(allowed)}"
                    ),
                )
            )
        direction = decision.get("remedy_direction")
        if direction is not None:
            expected = EXPECTED_DIRECTION.get(rule)
            if expected is not None and direction != expected:
                issues.append(
                    DefensibilityFailure(
                        kind="direction_mismatch",
                        epoch=epoch,
                        rule=rule,
                        detail=(
                            f"decision at epoch {epoch} cites {rule} with "
                            f"remedy_direction={direction!r}, expected {expected!r}"
                        ),
                    )
                )

    # Flag trivially-tripped rules: justification clearly indicates padding
    justification = str(decision.get("justification", "")).lower()
    if any(tok in justification for tok in ("pad log", "deliberate trip", "cosmetic")):
        issues.append(
            DefensibilityFailure(
                kind="pathological_pad",
                epoch=epoch,
                rule=",".join(cites) or "?",
                detail=f"justification at epoch {epoch} suggests log padding",
            )
        )

    return issues


def sample_decisions(
    decisions: list[dict], *, sample_size: int, seed: int = 42
) -> list[dict]:
    if sample_size >= len(decisions) or sample_size <= 0:
        return list(decisions)
    rng = random.Random(seed)
    return rng.sample(decisions, sample_size)


def audit_defensibility(
    decisions: list[dict],
    *,
    sample_size: int = 10,
    seed: int = 42,
) -> list[DefensibilityFailure]:
    """Run defensibility checks on a sampled subset of decisions."""
    sample = sample_decisions(decisions, sample_size=sample_size, seed=seed)
    failures: list[DefensibilityFailure] = []
    for d in sample:
        failures.extend(check_decision(d))
    return failures
