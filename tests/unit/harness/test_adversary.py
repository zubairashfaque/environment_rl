"""Tests for PromptAdversary."""

import pytest

from env_rl.harness.prompt_tuning.adversary import PromptAdversary
from env_rl.harness.prompt_tuning.scenarios import Scenario
from tests.unit.harness._stubs import StubOpenAIClient


def _adversary_response(n_scenarios: int = 2) -> dict:
    """Build a well-formed JSON response matching the adversary schema."""
    scenarios = []
    for i in range(n_scenarios):
        scenarios.append({
            "name": f"edge_case_{i}",
            "description": f"tricky case {i}",
            "top_rule": "R7",
            "fired_rules": ["R7", "R1"],
            "diagnostic_values": {
                "max_layer_grad_norm": 12.0 + i,
                "min_layer_grad_norm": 0.5,
                "dead_relu_fraction": 0.1,
                "train_loss": 2.0,
                "val_loss": 2.0,
                "train_acc": 0.3,
                "update_to_param_ratio": 0.05,
                "grad_noise_scale": 80.0,
            },
            "expected_event_type": "hyperparameter_change",
            "expected_remedy_direction": "decrease_lr",
            "current_lr": 0.3,
            "current_batch_size": 128,
        })
    return {"scenarios": scenarios}


def test_adversary_generates_scenario_list() -> None:
    client = StubOpenAIClient(responses=[_adversary_response(n_scenarios=2)])
    adv = PromptAdversary(client=client, model="gpt-4o-mini")
    candidates = adv.generate_candidates("SYSTEM PROMPT")
    assert len(candidates) == 2
    assert all(isinstance(c, Scenario) for c in candidates)
    assert candidates[0].top_rule == "R7"
    assert candidates[0].all_fired["R7"] is True
    assert candidates[0].all_fired["R1"] is True


def test_adversary_scenario_has_3_epoch_history() -> None:
    client = StubOpenAIClient(responses=[_adversary_response(n_scenarios=1)])
    adv = PromptAdversary(client=client)
    candidates = adv.generate_candidates("SYSTEM")
    assert len(candidates[0].metrics_history) == 3


def test_adversary_handles_empty_response_gracefully() -> None:
    client = StubOpenAIClient(responses=[{"scenarios": []}])
    adv = PromptAdversary(client=client)
    assert adv.generate_candidates("SYSTEM") == []


def test_adversary_keep_failures_returns_failed_scenarios_only() -> None:
    from env_rl.harness.prompt_tuning.scenarios import ScenarioResult, SCENARIO_SUITE
    failed = ScenarioResult(
        scenario=SCENARIO_SUITE[0], llm_decision={},
        passed=False, failure_reasons=["x"],
    )
    passed = ScenarioResult(
        scenario=SCENARIO_SUITE[1], llm_decision={},
        passed=True, failure_reasons=[],
    )
    kept = PromptAdversary.keep_failures([failed, passed])
    assert kept == [SCENARIO_SUITE[0]]
