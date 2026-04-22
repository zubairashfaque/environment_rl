"""Tests for human-review I/O and scenario conversion."""

import json

import pytest

from env_rl.harness.human_review import (
    HumanReview,
    append_review,
    load_all_reviews,
    load_reviews,
    reviews_to_scenarios,
)


def _gold_review(epoch: int = 3, rule: str = "R7") -> HumanReview:
    return HumanReview(
        attempt_dir="attempt_01",
        epoch=epoch,
        decision={
            "kind": "decision",
            "cites": [rule],
            "event_type": "hyperparameter_change",
            "remedy_direction": "decrease_lr",
        },
        verdict="gold",
        notes="perfect R7 response",
        reviewer="zubair",
    )


def test_append_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "decision_review.jsonl"
    r1 = _gold_review(epoch=3, rule="R7")
    r2 = _gold_review(epoch=5, rule="R5")
    append_review(path, r1)
    append_review(path, r2)
    loaded = load_reviews(path)
    assert len(loaded) == 2
    assert loaded[0].epoch == 3
    assert loaded[1].decision["cites"] == ["R5"]


def test_load_reviews_missing_file_returns_empty(tmp_path) -> None:
    assert load_reviews(tmp_path / "nope.jsonl") == []


def test_load_all_reviews_walks_attempt_dirs(tmp_path) -> None:
    base = tmp_path / "runs"
    for i in (1, 2):
        attempt = base / f"attempt_{i:02d}"
        attempt.mkdir(parents=True)
        append_review(attempt / "decision_review.jsonl", _gold_review(epoch=i, rule="R7"))
    all_reviews = load_all_reviews(base)
    assert len(all_reviews) == 2


def test_reviews_to_scenarios_converts_only_gold() -> None:
    gold = _gold_review(rule="R4")
    bad = HumanReview(
        attempt_dir="a", epoch=1,
        decision={"cites": ["R1"], "event_type": "rule_triggered_no_action",
                  "remedy_direction": "none"},
        verdict="bad",
    )
    skipped = HumanReview(
        attempt_dir="a", epoch=2,
        decision={"cites": ["R2"]}, verdict="skip",
    )
    scenarios = reviews_to_scenarios([gold, bad, skipped])
    assert len(scenarios) == 1
    assert scenarios[0].top_rule == "R4"
    assert "hyperparameter_change" in scenarios[0].expected.event_types


def test_reviews_to_scenarios_skips_decisions_with_no_cites() -> None:
    reviewless = HumanReview(
        attempt_dir="a", epoch=1,
        decision={"cites": []}, verdict="gold",
    )
    assert reviews_to_scenarios([reviewless]) == []
