"""Tests for Stage 6 features: scoreboard, decision cache, ensemble voting."""

import json

import pytest

from env_rl.harness.decision_cache import DecisionCache, fingerprint
from env_rl.harness.ensemble import (
    EnsembleResult,
    ensemble_decide,
    should_ensemble,
)
from env_rl.harness.prompt_tuning.scoreboard import TechniqueScoreboard
from tests.unit.harness._stubs import StubOpenAIClient


# ============================================================================
# Scoreboard
# ============================================================================


def test_scoreboard_records_and_persists(tmp_path) -> None:
    sb = TechniqueScoreboard(path=tmp_path / "sb.json")
    sb.record("one_shot", "win")
    sb.record("one_shot", "win")
    sb.record("one_shot", "loss")
    sb.record("few_shot", "tie")

    # New instance loads from disk
    sb2 = TechniqueScoreboard(path=tmp_path / "sb.json")
    assert sb2.win_rate("one_shot") == pytest.approx(2 / 3)
    assert sb2.win_rate("few_shot") == pytest.approx(0.5)  # no wins/losses


def test_scoreboard_rejects_bad_outcome(tmp_path) -> None:
    sb = TechniqueScoreboard(path=tmp_path / "sb.json")
    with pytest.raises(ValueError):
        sb.record("one_shot", "winnerrrr")


def test_scoreboard_summary_shape(tmp_path) -> None:
    sb = TechniqueScoreboard(path=tmp_path / "sb.json")
    sb.record("cot", "win")
    sb.record("cot", "loss")
    summary = sb.summary()
    assert "cot" in summary
    assert summary["cot"]["total"] == 2
    assert summary["cot"]["win_rate"] == pytest.approx(0.5)


# ============================================================================
# Decision cache
# ============================================================================


def test_fingerprint_stable_across_calls() -> None:
    a = fingerprint(top_rule="R7", all_fired={"R7": True, "R1": True},
                    current_lr=0.1, current_batch_size=128)
    b = fingerprint(top_rule="R7", all_fired={"R1": True, "R7": True},
                    current_lr=0.1, current_batch_size=128)
    assert a == b


def test_fingerprint_changes_with_lr() -> None:
    a = fingerprint(top_rule="R7", all_fired={"R7": True},
                    current_lr=0.1, current_batch_size=128)
    b = fingerprint(top_rule="R7", all_fired={"R7": True},
                    current_lr=0.05, current_batch_size=128)
    assert a != b


def test_decision_cache_hit_miss() -> None:
    cache = DecisionCache()
    fp = "abcd1234"
    assert cache.get(fp) is None
    cache.put(fp, {"decision": "x"})
    assert cache.get(fp) == {"decision": "x"}
    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_decision_cache_eviction() -> None:
    cache = DecisionCache(max_size=3)
    for i in range(5):
        cache.put(f"fp_{i}", {"i": i})
    # First two should have been evicted
    assert cache.get("fp_0") is None
    assert cache.get("fp_1") is None
    assert cache.get("fp_4") == {"i": 4}


# ============================================================================
# Ensemble
# ============================================================================


def test_should_ensemble_gates_on_stability_rules() -> None:
    assert should_ensemble("R7") is True
    assert should_ensemble("R6") is True
    assert should_ensemble("R1") is False
    assert should_ensemble("R5") is False


def test_ensemble_decide_majority_vote() -> None:
    # Two R7/decrease_lr, one R1/increase_lr → R7/decrease_lr wins
    samples = [
        {"event_type": "hyperparameter_change", "cites": ["R7"],
         "justification": "a", "remedy_direction": "decrease_lr",
         "remedy_params": {"lr_new": 0.01, "edit_op": "none", "edit_to": "none"}},
        {"event_type": "hyperparameter_change", "cites": ["R7"],
         "justification": "b", "remedy_direction": "decrease_lr",
         "remedy_params": {"lr_new": 0.02, "edit_op": "none", "edit_to": "none"}},
        {"event_type": "hyperparameter_change", "cites": ["R1"],
         "justification": "c", "remedy_direction": "increase_lr",
         "remedy_params": {"lr_new": 0.5, "edit_op": "none", "edit_to": "none"}},
    ]
    client = StubOpenAIClient(responses=samples)
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {}}}
    result = ensemble_decide(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "go"}],
        json_schema_response_format=fmt,
        n_samples=3, temperature=0.7,
    )
    assert result.majority_decision["cites"] == ["R7"]
    assert result.majority_decision["remedy_direction"] == "decrease_lr"
    assert result.agreement == pytest.approx(2 / 3)


def test_ensemble_decide_unanimous() -> None:
    same = {
        "event_type": "hyperparameter_change", "cites": ["R7"],
        "justification": "", "remedy_direction": "decrease_lr",
        "remedy_params": {"lr_new": 0.01, "edit_op": "none", "edit_to": "none"},
    }
    client = StubOpenAIClient(responses=[same, dict(same), dict(same)])
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {}}}
    result = ensemble_decide(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "go"}],
        json_schema_response_format=fmt,
        n_samples=3,
    )
    assert result.agreement == 1.0
