"""Tests for the unified AgentTracer."""

import json

import pytest

from env_rl.harness.agent_trace import NULL_TRACER, AgentTracer


def test_null_tracer_is_noop() -> None:
    assert NULL_TRACER.enabled is False
    NULL_TRACER.record(agent="x", action="y")  # must not raise
    assert NULL_TRACER.event_count == 0


def test_tracer_writes_jsonl(tmp_path) -> None:
    t = AgentTracer(tmp_path / "agent_trace.jsonl")
    t.record(agent="decision_llm", action="api_call", duration_ms=12.3,
             input_summary={"a": 1}, output_summary={"b": 2},
             token_cost={"total_tokens": 100}, model="gpt-4o-mini")
    t.record(agent="tuner", action="propose_edit", duration_ms=0.5)
    lines = (tmp_path / "agent_trace.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["agent"] == "decision_llm"
    assert first["action"] == "api_call"
    assert first["duration_ms"] == 12.3
    assert first["token_cost"]["total_tokens"] == 100
    assert first["model"] == "gpt-4o-mini"


def test_tracer_set_attempt_index(tmp_path) -> None:
    t = AgentTracer(tmp_path / "agent_trace.jsonl")
    t.set_attempt_index(3)
    t.record(agent="x", action="y")
    rec = json.loads((tmp_path / "agent_trace.jsonl").read_text().splitlines()[0])
    assert rec["attempt_index"] == 3


def test_tracer_timed_context_measures_duration(tmp_path) -> None:
    import time
    t = AgentTracer(tmp_path / "agent_trace.jsonl")
    with t.timed(agent="x", action="y") as out:
        time.sleep(0.01)
        out["output_summary"] = {"foo": "bar"}
    rec = json.loads((tmp_path / "agent_trace.jsonl").read_text().splitlines()[0])
    assert rec["duration_ms"] >= 10.0
    assert rec["output_summary"] == {"foo": "bar"}


def test_tracer_disabled_flag(tmp_path) -> None:
    t = AgentTracer(tmp_path / "agent_trace.jsonl", enabled=False)
    t.record(agent="x", action="y")
    assert t.event_count == 0
    assert t.enabled is False
    assert not (tmp_path / "agent_trace.jsonl").exists()
