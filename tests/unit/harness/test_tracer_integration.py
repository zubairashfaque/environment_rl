"""Verify that tracer records events from every agent class."""

import json

import pytest

from env_rl.harness.agent_trace import AgentTracer
from env_rl.harness.policy import OpenAIDecisionPolicy
from env_rl.harness.prompt_tuning.meta_loop import MetaLoop
from env_rl.harness.prompt_tuning.scenarios import SCENARIO_SUITE
from tests.unit.harness._stubs import StubOpenAIClient


def _r7_response() -> dict:
    return {
        "event_type": "hyperparameter_change",
        "cites": ["R7"],
        "justification": "drop lr",
        "remedy_direction": "decrease_lr",
        "remedy_params": {"lr_new": 0.03, "edit_op": "none", "edit_to": "none"},
    }


def test_policy_records_api_call_and_cache_hit(tmp_path) -> None:
    tracer = AgentTracer(tmp_path / "agent_trace.jsonl")
    client = StubOpenAIClient(responses=[_r7_response()])
    policy = OpenAIDecisionPolicy(
        client=client, model="gpt-4o-mini",
        system_prompt="SYSTEM", enable_cache=True,
        tracer=tracer,
    )
    ctx = dict(
        top_rule="R7", all_fired={"R7": True},
        metrics={"epoch": 3}, epoch=3,
        current_lr=0.3, current_batch_size=128, recent_history=[],
    )
    policy.decide(**ctx)      # api_call event
    policy.decide(**ctx)      # cache_hit event

    events = [
        json.loads(line) for line in
        (tmp_path / "agent_trace.jsonl").read_text().splitlines() if line.strip()
    ]
    actions = [e["action"] for e in events]
    assert "api_call" in actions
    assert "cache_hit" in actions
    agents = {e["agent"] for e in events}
    assert agents == {"decision_llm"}


def test_meta_loop_records_tuner_tester_judge(tmp_path) -> None:
    tracer = AgentTracer(tmp_path / "agent_trace.jsonl")
    n = len(SCENARIO_SUITE)
    client = StubOpenAIClient(responses=[_r7_response()] * (n * 3))
    ml = MetaLoop(
        base_dir=tmp_path, initial_prompt="INITIAL",
        tester_client=client, tracer=tracer,
    )
    ml.step(
        attempt_index=1,
        violations=[{"kind": "unresolved_deferral", "rule": "R1", "epoch": 3}],
    )
    events = [
        json.loads(line) for line in
        (tmp_path / "agent_trace.jsonl").read_text().splitlines() if line.strip()
    ]
    agents = {e["agent"] for e in events}
    # Expect at least tuner, tester, judge, scoreboard
    assert "tuner" in agents
    assert "tester" in agents
    assert "judge" in agents
    assert "scoreboard" in agents
