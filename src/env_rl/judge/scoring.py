"""Judge step 11: two-axis decoupled scoring.

The two axes are not tradeable: hard-fails in steps 1–7 zero both; violations
in steps 8–9 reduce process only; low test accuracy reduces accuracy only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scores:
    accuracy_score: float
    process_score: float
    hard_fail: bool
    test_accuracy: float
    violations: int
    total_decisions: int


def accuracy_score(
    test_accuracy: float,
    target_acc: float,
    saturation_ceiling: float = 0.95,
) -> float:
    """Continuous, monotonically increasing. Hits 1.0 at ``target_acc`` and
    stays there (also flagged as "saturating at a ceiling above the target"
    in the spec — ``saturation_ceiling`` is reserved for future expansion).
    """
    if test_accuracy >= target_acc:
        return 1.0
    if test_accuracy <= 0.0:
        return 0.0
    return max(0.0, test_accuracy / target_acc)


def process_score(violations: int, total_decisions: int) -> float:
    """``1 - violations / total_decisions``, bounded in [0, 1].

    Note: the denominator-gaming caveat is documented in README and plan §12.
    """
    if total_decisions <= 0:
        return 1.0  # see denominator-gaming note
    return max(0.0, min(1.0, 1.0 - violations / total_decisions))


def compute_scores(
    *,
    hard_fail: bool,
    test_accuracy: float,
    target_acc: float,
    violations: int,
    total_decisions: int,
    saturation_ceiling: float = 0.95,
) -> Scores:
    if hard_fail:
        return Scores(
            accuracy_score=0.0,
            process_score=0.0,
            hard_fail=True,
            test_accuracy=test_accuracy,
            violations=violations,
            total_decisions=total_decisions,
        )
    return Scores(
        accuracy_score=accuracy_score(test_accuracy, target_acc, saturation_ceiling),
        process_score=process_score(violations, total_decisions),
        hard_fail=False,
        test_accuracy=test_accuracy,
        violations=violations,
        total_decisions=total_decisions,
    )
