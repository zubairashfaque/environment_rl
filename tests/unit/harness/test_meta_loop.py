"""Tests for the MetaLoop orchestrator."""

import json

import pytest

from env_rl.harness.prompt_tuning.meta_loop import MetaLoop
from env_rl.harness.prompt_tuning.scenarios import SCENARIO_SUITE
from tests.unit.harness._stubs import StubOpenAIClient


def _good_response(rule: str = "R7") -> dict:
    """Correctly-shaped decision that passes the R7 scenario."""
    return {
        "event_type": "hyperparameter_change",
        "cites": [rule],
        "justification": "drop lr",
        "remedy_direction": "decrease_lr",
        "remedy_params": {"lr_new": 0.01, "edit_op": "none", "edit_to": "none"},
    }


def _bad_response() -> dict:
    """Decision that will fail most scenarios."""
    return {
        "event_type": "rule_triggered_no_action",
        "cites": ["R1"],
        "justification": "waiting",
        "remedy_direction": "none",
        "remedy_params": {"lr_new": 0.05, "edit_op": "none", "edit_to": "none"},
    }


def test_meta_loop_persists_initial_version(tmp_path) -> None:
    # Initial evaluation: every scenario gets a generic response
    client = StubOpenAIClient(responses=[_good_response("R7")] * len(SCENARIO_SUITE))
    ml = MetaLoop(
        base_dir=tmp_path, initial_prompt="INITIAL",
        tester_client=client, tester_model="gpt-4o-mini",
    )
    assert ml.champion_version == 0
    assert ml.champion_prompt == "INITIAL"
    assert (tmp_path / "prompts" / "v000.txt").exists()


def test_meta_loop_step_noop_when_no_violations(tmp_path) -> None:
    client = StubOpenAIClient(responses=[_good_response()] * (len(SCENARIO_SUITE) * 3))
    ml = MetaLoop(
        base_dir=tmp_path, initial_prompt="INITIAL",
        tester_client=client,
    )
    iteration = ml.step(attempt_index=1, violations=[])
    assert iteration.winner == "tie"
    assert iteration.proposed_edit_technique == "baseline"
    # champion unchanged
    assert ml.champion_version == 0


def test_meta_loop_step_proposes_and_evaluates_edit(tmp_path) -> None:
    # Both prompts pass all scenarios equally (same responses)
    n = len(SCENARIO_SUITE)
    responses = [_good_response()] * (n * 4)
    client = StubOpenAIClient(responses=responses)
    ml = MetaLoop(base_dir=tmp_path, initial_prompt="INITIAL", tester_client=client)
    iteration = ml.step(
        attempt_index=1,
        violations=[{"kind": "unresolved_deferral", "rule": "R1", "epoch": 3}],
    )
    # A new version was persisted (v001.txt), even if judge ties
    assert (tmp_path / "prompts" / "v001.txt").exists()
    log = json.loads((tmp_path / "meta_loop_log.json").read_text())
    assert log["iterations"][-1]["proposed_edit_technique"] == "negative_constraint"


def test_meta_loop_promotes_winning_edit(tmp_path) -> None:
    """If the new prompt scores strictly better, it becomes champion."""
    n = len(SCENARIO_SUITE)
    # First n responses (initial): half right, half wrong
    initial = [_good_response() for _ in range(n // 2)] + [_bad_response()] * (n - n // 2)
    # Next n responses (new prompt): all right
    new_round = [_good_response()] * n
    client = StubOpenAIClient(responses=initial + new_round)

    ml = MetaLoop(base_dir=tmp_path, initial_prompt="INITIAL", tester_client=client)
    v0_rate = ml._versions[0].scenario_pass_rate
    assert v0_rate < 1.0  # initial prompt is imperfect

    iteration = ml.step(
        attempt_index=1,
        violations=[
            {"kind": "unresolved_deferral", "rule": "R1", "epoch": 3},
            {"kind": "unresolved_deferral", "rule": "R1", "epoch": 4},
        ],
    )
    assert iteration.winner == "new"
    assert ml.champion_version == 1
