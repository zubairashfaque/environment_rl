import pytest

from env_rl.judge.scoring import (
    accuracy_score,
    compute_scores,
    process_score,
)


def test_accuracy_score_at_target_is_one() -> None:
    assert accuracy_score(0.92, target_acc=0.92) == 1.0


def test_accuracy_score_above_target_saturates_at_one() -> None:
    assert accuracy_score(0.97, target_acc=0.92) == 1.0


def test_accuracy_score_below_target_is_monotone_less_than_one() -> None:
    s = accuracy_score(0.80, target_acc=0.92)
    assert s < 1.0
    assert s == pytest.approx(0.80 / 0.92, rel=1e-6)


def test_accuracy_score_zero_for_zero_accuracy() -> None:
    assert accuracy_score(0.0, target_acc=0.92) == 0.0


def test_process_score_one_for_zero_violations() -> None:
    assert process_score(0, 10) == 1.0


def test_process_score_40_decisions_1_violation_is_0_975() -> None:
    assert process_score(1, 40) == pytest.approx(0.975, rel=1e-6)


def test_process_score_denominator_gaming_saturates_to_one() -> None:
    # documented behavior per plan §12
    assert process_score(0, 0) == 1.0


def test_hard_fail_zeros_both_scores() -> None:
    scores = compute_scores(
        hard_fail=True,
        test_accuracy=0.95,
        target_acc=0.92,
        violations=0,
        total_decisions=10,
    )
    assert scores.accuracy_score == 0.0
    assert scores.process_score == 0.0
    assert scores.hard_fail is True


def test_compute_scores_scoring_path() -> None:
    scores = compute_scores(
        hard_fail=False,
        test_accuracy=0.92,
        target_acc=0.92,
        violations=1,
        total_decisions=40,
    )
    assert scores.accuracy_score == 1.0
    assert scores.process_score == pytest.approx(0.975, rel=1e-6)
