import pytest

from env_rl.harness.policy import (
    OpenAIDecisionPolicy,
    ScriptedDecisionPolicy,
)
from tests.unit.harness._stubs import StubOpenAIClient


def _ctx(**overrides):
    defaults = dict(
        top_rule="R7",
        all_fired={f"R{i}": (i == 7) for i in range(1, 8)},
        metrics={
            "epoch": 5,
            "max_layer_grad_norm": 14.2,
            "min_layer_grad_norm": 0.01,
            "dead_relu_fraction": 0.1,
            "update_to_param_ratio": 5e-2,
            "grad_noise_scale": 300.0,
            "val_loss": 2.0,
            "train_loss": 2.0,
            "val_acc": 0.2,
            "train_acc": 0.25,
        },
        epoch=5,
        current_lr=0.3,
        current_batch_size=128,
        recent_history=[{"epoch": 5}],
    )
    defaults.update(overrides)
    return defaults


def test_scripted_policy_r7_drops_lr_by_ten() -> None:
    policy = ScriptedDecisionPolicy()
    d = policy.decide(**_ctx(top_rule="R7"))
    assert d.event_type == "hyperparameter_change"
    assert d.cites == ["R7"]
    assert d.remedy_direction == "decrease_lr"
    assert d.remedy_params["lr_new"] == pytest.approx(0.03)


def test_scripted_policy_r5_is_architecture_change() -> None:
    policy = ScriptedDecisionPolicy()
    d = policy.decide(**_ctx(top_rule="R5"))
    assert d.event_type == "architecture_change"
    assert d.remedy_direction == "swap_activation"


def test_openai_policy_parses_decision_from_structured_output() -> None:
    client = StubOpenAIClient(
        responses=[
            {
                "event_type": "hyperparameter_change",
                "cites": ["R7"],
                "justification": "max_layer_grad_norm 14.2 for 3 epochs; drop LR",
                "remedy_direction": "decrease_lr",
                "remedy_params": {"lr_new": 0.03, "edit_op": "none", "edit_to": ""},
            }
        ]
    )
    policy = OpenAIDecisionPolicy(client=client, model="gpt-4o-mini")
    d = policy.decide(**_ctx(top_rule="R7"))
    assert d.event_type == "hyperparameter_change"
    assert d.cites == ["R7"]
    assert d.remedy_direction == "decrease_lr"
    assert d.remedy_params["lr_new"] == pytest.approx(0.03)
    # one call was issued with the right model + response_format
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"]["type"] == "json_schema"


def test_openai_policy_drops_lr_new_when_same_as_current() -> None:
    client = StubOpenAIClient(
        responses=[
            {
                "event_type": "rule_triggered_no_action",
                "cites": ["R1"],
                "justification": "watch one more epoch",
                "remedy_direction": "none",
                "remedy_params": {"lr_new": 0.3, "edit_op": "none", "edit_to": ""},
            }
        ]
    )
    policy = OpenAIDecisionPolicy(client=client, model="gpt-4o-mini")
    d = policy.decide(**_ctx(top_rule="R1", current_lr=0.3))
    assert "lr_new" not in d.remedy_params  # same as current → no change


def test_openai_policy_parses_architecture_edit() -> None:
    client = StubOpenAIClient(
        responses=[
            {
                "event_type": "architecture_change",
                "cites": ["R5"],
                "justification": "dead relu >0.4",
                "remedy_direction": "swap_activation",
                "remedy_params": {
                    "lr_new": 0.3,
                    "edit_op": "swap_activation",
                    "edit_to": "leaky_relu",
                },
            }
        ]
    )
    policy = OpenAIDecisionPolicy(client=client, model="gpt-4o-mini")
    d = policy.decide(**_ctx(top_rule="R5", current_lr=0.3))
    assert d.event_type == "architecture_change"
    assert d.remedy_params["edit"] == {"op": "swap_activation", "to": "leaky_relu"}


def test_openai_policy_carries_prior_attempts_into_system_prompt() -> None:
    from env_rl.harness.prompt import AttemptSummary

    prior = [
        AttemptSummary(
            attempt_index=1,
            accuracy_score=0.2,
            process_score=0.5,
            test_accuracy=0.15,
            total_decisions=4,
            violations=2,
            violation_summary=[
                {"kind": "event_type_mismatch", "rule": "R4", "epoch": 5, "detail": "x"}
            ],
        )
    ]
    client = StubOpenAIClient(responses=[{"event_type": "hyperparameter_change",
                                          "cites": ["R7"],
                                          "justification": "x",
                                          "remedy_direction": "decrease_lr",
                                          "remedy_params": {"lr_new": 0.03, "edit_op": "none", "edit_to": ""}}])
    policy = OpenAIDecisionPolicy(client=client, model="gpt-4o-mini", prior_attempts=prior)
    assert "PRIOR ATTEMPTS" in policy.system_prompt
    assert "event_type_mismatch" in policy.system_prompt
    policy.decide(**_ctx(top_rule="R7"))
    # and the prompt was passed to the call
    system_msg = client.calls[0]["messages"][0]["content"]
    assert "PRIOR ATTEMPTS" in system_msg
