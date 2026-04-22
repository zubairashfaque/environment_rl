"""Tests for the prompt-evolution agents (Tuner, Tester, Judge)."""

import json

import pytest

from env_rl.harness.prompt_tuning import (
    PROMPT_TECHNIQUES,
    PromptEdit,
    PromptJudge,
    PromptTester,
    PromptTuner,
    SCENARIO_SUITE,
)
from env_rl.harness.prompt_tuning.judge import PromptJudgment
from env_rl.harness.prompt_tuning.scenarios import ScenarioResult
from tests.unit.harness._stubs import StubOpenAIClient


# ============================================================================
# Tuner
# ============================================================================


def test_tuner_baseline_when_no_violations() -> None:
    edit = PromptTuner().propose_edit(violations=[], attempt_index=1)
    assert edit.technique == "baseline"
    assert edit.addition == ""


def test_tuner_negative_constraint_for_few_violations() -> None:
    edit = PromptTuner().propose_edit(
        violations=[{"kind": "unresolved_deferral", "rule": "R1", "epoch": 3}],
        attempt_index=1,
    )
    assert edit.technique == "negative_constraint"
    assert "R1" in edit.addition


def test_tuner_one_shot_for_moderate_violations() -> None:
    edit = PromptTuner().propose_edit(
        violations=[
            {"kind": "direction_mismatch", "rule": "R7", "epoch": i}
            for i in range(4)
        ],
        attempt_index=1,
    )
    assert edit.technique == "one_shot"
    assert "R7" in edit.addition


def test_tuner_few_shot_for_many_violations() -> None:
    # 8 total, 3 distinct rules so the few_shot path is hit
    v = []
    for r in ("R7", "R5", "R4"):
        v.extend([{"kind": "unresolved_deferral", "rule": r, "epoch": i} for i in range(3)])
    edit = PromptTuner().propose_edit(violations=v[:9], attempt_index=1)
    assert edit.technique == "few_shot"


def test_tuner_escalates_to_role_play_on_repeat_attempts() -> None:
    v = [
        {"kind": "unresolved_deferral", "rule": "R1", "epoch": i}
        for i in range(15)
    ]
    edit = PromptTuner().propose_edit(violations=v, attempt_index=3)
    assert edit.technique in ("role_play", "chain_of_thought")


def test_prompt_edit_apply_appends_addition() -> None:
    base = "Original prompt content."
    edit = PromptEdit(technique="negative_constraint", rationale="", addition="New rule: X")
    result = edit.apply(base)
    assert base in result
    assert "New rule: X" in result


def test_prompt_edit_apply_no_op_on_empty_addition() -> None:
    base = "Original."
    assert PromptEdit("baseline", "", "").apply(base) == base


def test_all_techniques_registered() -> None:
    for t in ("baseline", "negative_constraint", "one_shot",
              "few_shot", "chain_of_thought", "role_play"):
        assert t in PROMPT_TECHNIQUES


# ============================================================================
# Tester
# ============================================================================


def _stub_response(event_type: str, rule: str, direction: str,
                   edit_op: str = "none", edit_to: str = "none",
                   lr_new: float = 0.05) -> dict:
    return {
        "event_type": event_type,
        "cites": [rule],
        "justification": "test",
        "remedy_direction": direction,
        "remedy_params": {"lr_new": lr_new, "edit_op": edit_op, "edit_to": edit_to},
    }


def test_tester_passes_correct_r7_decision() -> None:
    # Find the R7 scenario
    r7 = next(s for s in SCENARIO_SUITE if s.name == "r7_exploding_drop_lr")
    client = StubOpenAIClient(responses=[
        _stub_response("hyperparameter_change", "R7", "decrease_lr", lr_new=0.03),
    ])
    tester = PromptTester(client=client, model="gpt-4o-mini")
    results = tester.run_suite("SYSTEM PROMPT", scenarios=[r7])
    assert results[0].passed


def test_tester_fails_wrong_event_type() -> None:
    r5 = next(s for s in SCENARIO_SUITE if s.name == "r5_dead_relu_swap_activation")
    client = StubOpenAIClient(responses=[
        # event_type should be architecture_change; giving hyperparameter_change
        _stub_response("hyperparameter_change", "R5", "swap_activation",
                       edit_op="swap_activation", edit_to="leaky_relu"),
    ])
    tester = PromptTester(client=client, model="gpt-4o-mini")
    results = tester.run_suite("SYSTEM PROMPT", scenarios=[r5])
    assert not results[0].passed
    assert any("event_type" in r for r in results[0].failure_reasons)


def test_tester_fails_wrong_edit_op() -> None:
    r4 = next(s for s in SCENARIO_SUITE if s.name == "r4_capacity_add_block")
    client = StubOpenAIClient(responses=[
        # Correct class and direction, but edit_op not add_block
        _stub_response("architecture_change", "R4", "add_capacity",
                       edit_op="swap_activation", edit_to="leaky_relu"),
    ])
    tester = PromptTester(client=client, model="gpt-4o-mini")
    results = tester.run_suite("SYSTEM PROMPT", scenarios=[r4])
    assert not results[0].passed
    assert any("edit_op" in r for r in results[0].failure_reasons)


# ============================================================================
# Judge
# ============================================================================


def _mk_result(name: str, passed: bool) -> ScenarioResult:
    scn = next(s for s in SCENARIO_SUITE if s.name == name)
    return ScenarioResult(
        scenario=scn, llm_decision={}, passed=passed, failure_reasons=[]
    )


def test_judge_picks_new_when_it_strictly_improves() -> None:
    j = PromptJudge()
    old_results = [_mk_result("r7_exploding_drop_lr", True),
                   _mk_result("r5_dead_relu_swap_activation", False)]
    new_results = [_mk_result("r7_exploding_drop_lr", True),
                   _mk_result("r5_dead_relu_swap_activation", True)]
    verdict = j.compare(old_prompt="short", new_prompt="short+edit",
                        old_results=old_results, new_results=new_results)
    assert verdict.winner == "new"
    assert "r5_dead_relu_swap_activation" in verdict.improvements


def test_judge_picks_old_when_new_regresses() -> None:
    j = PromptJudge()
    old_results = [_mk_result("r7_exploding_drop_lr", True),
                   _mk_result("r5_dead_relu_swap_activation", True)]
    new_results = [_mk_result("r7_exploding_drop_lr", True),
                   _mk_result("r5_dead_relu_swap_activation", False)]
    verdict = j.compare(old_prompt="short", new_prompt="longer_prompt",
                        old_results=old_results, new_results=new_results)
    assert verdict.winner == "old"
    assert "r5_dead_relu_swap_activation" in verdict.regressions


def test_judge_prefers_shorter_prompt_on_tie() -> None:
    j = PromptJudge()
    results = [_mk_result("r7_exploding_drop_lr", True)]
    # Same pass rate but new_prompt much longer → old wins via length penalty
    verdict = j.compare(
        old_prompt="x" * 100,
        new_prompt="x" * 10_000,
        old_results=results,
        new_results=results,
    )
    assert verdict.winner == "old"
