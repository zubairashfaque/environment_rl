from env_rl.harness.prompt import (
    AttemptSummary,
    build_decision_messages,
    build_iterative_system_prompt,
    playbook_text,
)


def test_playbook_text_contains_all_seven_rules() -> None:
    text = playbook_text()
    for rule_id in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert rule_id in text


def test_system_prompt_without_priors_contains_response_rules() -> None:
    sp = build_iterative_system_prompt(prior_attempts=None)
    assert "Conflict precedence" in sp
    assert "stability > capacity > tuning > process" in sp
    assert "R7" in sp  # playbook injected by default


def test_system_prompt_includes_prior_attempt_feedback() -> None:
    prior = [
        AttemptSummary(
            attempt_index=1,
            accuracy_score=0.2,
            process_score=0.5,
            test_accuracy=0.18,
            total_decisions=4,
            violations=2,
            violation_summary=[
                {
                    "kind": "event_type_mismatch",
                    "rule": "R4",
                    "epoch": 5,
                    "detail": "cited R4 with hyperparameter_change",
                }
            ],
        )
    ]
    sp = build_iterative_system_prompt(prior_attempts=prior)
    assert "PRIOR ATTEMPTS" in sp
    assert "Attempt 1" in sp
    assert "event_type_mismatch" in sp
    assert "avoid repeating" in sp.lower()


def test_decision_messages_contains_epoch_rule_and_metrics() -> None:
    messages = build_decision_messages(
        system_prompt="SYSTEM",
        epoch=7,
        top_rule="R7",
        all_fired={"R7": True, "R1": True},
        metrics={
            "epoch": 7,
            "train_loss": 1.5,
            "val_loss": 1.7,
            "max_layer_grad_norm": 14.2,
            "min_layer_grad_norm": 0.01,
            "dead_relu_fraction": 0.1,
            "update_to_param_ratio": 5e-2,
            "grad_noise_scale": 300.0,
            "val_acc": 0.3,
            "train_acc": 0.35,
        },
        current_lr=0.3,
        current_batch_size=128,
        recent_history=[{"epoch": 5}, {"epoch": 6}, {"epoch": 7}],
    )
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    u = messages[1]["content"]
    assert "Epoch 7" in u
    assert "R7" in u
    assert "R1" in u  # shown as fired; driver handles deferral in a separate call
    assert "lr = 0.3" in u
    assert "batch_size = 128" in u
    assert "max_layer_grad_norm = 14.2" in u
