from env_rl.judge.defensibility import audit_defensibility, check_decision


def test_r7_hyperparameter_change_is_defensible() -> None:
    d = {
        "epoch": 5,
        "event_type": "hyperparameter_change",
        "cites": ["R7"],
        "justification": "dropped lr",
    }
    assert check_decision(d) == []


def test_r4_hyperparameter_change_is_indefensible() -> None:
    # R4 (depth) is a capacity rule; hyperparameter_change is wrong class
    d = {
        "epoch": 5,
        "event_type": "hyperparameter_change",
        "cites": ["R4"],
        "justification": "",
    }
    failures = check_decision(d)
    assert any(f.kind == "event_type_mismatch" and f.rule == "R4" for f in failures)


def test_r7_with_wrong_remedy_direction_flagged() -> None:
    d = {
        "epoch": 5,
        "event_type": "hyperparameter_change",
        "cites": ["R7"],
        "justification": "increased lr",
        "remedy_direction": "increase_lr",
    }
    failures = check_decision(d)
    assert any(f.kind == "direction_mismatch" and f.rule == "R7" for f in failures)


def test_pathological_padding_justification_flagged() -> None:
    d = {
        "epoch": 2,
        "event_type": "hyperparameter_change",
        "cites": ["R2"],
        "justification": "deliberate trip to pad log",
    }
    failures = check_decision(d)
    assert any(f.kind == "pathological_pad" for f in failures)


def test_audit_defensibility_aggregates_failures() -> None:
    decisions = [
        {"epoch": 1, "event_type": "hyperparameter_change", "cites": ["R7"], "justification": "ok"},
        {"epoch": 2, "event_type": "hyperparameter_change", "cites": ["R4"], "justification": "bad class"},
    ]
    failures = audit_defensibility(decisions, sample_size=10, seed=0)
    assert len(failures) == 1
    assert failures[0].rule == "R4"
