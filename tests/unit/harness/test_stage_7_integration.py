"""Integration tests for Stage 7: Decision Cache + Ensemble + Scoreboard wiring."""

import json

import pytest

from env_rl.harness.policy import OpenAIDecisionPolicy
from env_rl.harness.prompt_tuning.meta_loop import MetaLoop
from env_rl.harness.prompt_tuning.scenarios import SCENARIO_SUITE
from env_rl.harness.prompt_tuning.scoreboard import TechniqueScoreboard
from tests.unit.harness._stubs import StubOpenAIClient


def _r7_response() -> dict:
    return {
        "event_type": "hyperparameter_change",
        "cites": ["R7"],
        "justification": "drop lr",
        "remedy_direction": "decrease_lr",
        "remedy_params": {"lr_new": 0.03, "edit_op": "none", "edit_to": "none"},
    }


# ============================================================================
# Unit A — decision cache integration
# ============================================================================


def test_policy_decision_cache_hit_avoids_api_call(tmp_path) -> None:
    client = StubOpenAIClient(responses=[_r7_response()])
    policy = OpenAIDecisionPolicy(
        client=client, model="gpt-4o-mini",
        system_prompt="SYSTEM",
        transcript_path=tmp_path / "transcript.jsonl",
        enable_cache=True,
    )
    ctx = dict(
        top_rule="R7",
        all_fired={"R7": True, **{f"R{i}": False for i in (1, 2, 3, 4, 5, 6)}},
        metrics={"epoch": 3, "max_layer_grad_norm": 14.0},
        epoch=3, current_lr=0.3, current_batch_size=128, recent_history=[],
    )
    d1 = policy.decide(**ctx)
    d2 = policy.decide(**ctx)
    # Only one API call (client has one canned response; second call would
    # blow up if cache miss)
    assert d1.cites == d2.cites == ["R7"]
    stats = policy.cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_policy_transcript_records_cache_hit(tmp_path) -> None:
    client = StubOpenAIClient(responses=[_r7_response()])
    policy = OpenAIDecisionPolicy(
        client=client, system_prompt="SYSTEM",
        transcript_path=tmp_path / "t.jsonl",
        enable_cache=True,
    )
    ctx = dict(
        top_rule="R7", all_fired={"R7": True},
        metrics={"epoch": 3}, epoch=3,
        current_lr=0.3, current_batch_size=128, recent_history=[],
    )
    policy.decide(**ctx)
    policy.decide(**ctx)  # second call is a cache hit
    lines = [
        json.loads(line) for line in
        (tmp_path / "t.jsonl").read_text().splitlines() if line.strip()
    ]
    calls = [l for l in lines if l["kind"] == "call"]
    assert len(calls) == 2
    assert calls[0]["cache_hit"] is False
    assert calls[1]["cache_hit"] is True


def test_policy_cache_disabled_means_every_call_goes_to_api() -> None:
    client = StubOpenAIClient(responses=[_r7_response(), _r7_response()])
    policy = OpenAIDecisionPolicy(
        client=client, system_prompt="SYSTEM", enable_cache=False,
    )
    ctx = dict(
        top_rule="R7", all_fired={"R7": True},
        metrics={"epoch": 3}, epoch=3,
        current_lr=0.3, current_batch_size=128, recent_history=[],
    )
    policy.decide(**ctx)
    policy.decide(**ctx)
    # Both responses were consumed — no caching
    assert len(client.calls) == 2
    assert policy.cache_stats == {"enabled": False}


# ============================================================================
# Unit B — ensemble voting integration
# ============================================================================


def test_policy_ensemble_samples_three_times_for_r7() -> None:
    # Three samples, all R7/decrease_lr (unanimous)
    client = StubOpenAIClient(responses=[_r7_response()] * 3)
    policy = OpenAIDecisionPolicy(
        client=client, system_prompt="SYSTEM",
        enable_cache=False, enable_ensemble=True,
    )
    ctx = dict(
        top_rule="R7", all_fired={"R7": True},
        metrics={"epoch": 3}, epoch=3,
        current_lr=0.3, current_batch_size=128, recent_history=[],
    )
    d = policy.decide(**ctx)
    assert d.cites == ["R7"]
    assert len(client.calls) == 3


def test_policy_ensemble_off_for_non_stability_rule() -> None:
    # R1 is tuning, not stability — ensemble should skip
    r1_response = {
        "event_type": "hyperparameter_change",
        "cites": ["R1"],
        "justification": "lr high",
        "remedy_direction": "decrease_lr",
        "remedy_params": {"lr_new": 0.01, "edit_op": "none", "edit_to": "none"},
    }
    client = StubOpenAIClient(responses=[r1_response])
    policy = OpenAIDecisionPolicy(
        client=client, system_prompt="SYSTEM",
        enable_cache=False, enable_ensemble=True,
    )
    ctx = dict(
        top_rule="R1", all_fired={"R1": True},
        metrics={"epoch": 3}, epoch=3,
        current_lr=0.3, current_batch_size=128, recent_history=[],
    )
    policy.decide(**ctx)
    # Just one call even with enable_ensemble=True
    assert len(client.calls) == 1


# ============================================================================
# Unit C — scoreboard integration in MetaLoop
# ============================================================================


def test_meta_loop_records_technique_outcome_in_scoreboard(tmp_path) -> None:
    n = len(SCENARIO_SUITE)
    # Initial eval all correct; then proposed edit eval also all correct -> tie
    client = StubOpenAIClient(responses=[_r7_response()] * (n * 4))
    ml = MetaLoop(
        base_dir=tmp_path, initial_prompt="INITIAL",
        tester_client=client,
    )
    ml.step(
        attempt_index=1,
        violations=[{"kind": "unresolved_deferral", "rule": "R1", "epoch": 3}],
    )
    summary = ml.scoreboard_summary
    assert "negative_constraint" in summary
    entry = summary["negative_constraint"]
    assert entry["total"] == 1
