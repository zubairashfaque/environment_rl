from env_rl.judge.coverage import audit_rule_coverage


def _r(epoch: int, **flags) -> dict:
    evals = {f"R{i}": False for i in range(1, 8)}
    evals.update(flags)
    return {"epoch": epoch, "evals": evals}


def _d(
    epoch: int, event_type: str, cites: list[str], justification: str = ""
) -> dict:
    return {
        "epoch": epoch,
        "event_type": event_type,
        "cites": cites,
        "justification": justification,
    }


def test_coverage_no_violations_when_rule_matched_in_window() -> None:
    rule_evals = [_r(0), _r(1), _r(2)]
    rule_evals[2]["evals"]["R7"] = True
    decisions = [_d(3, "hyperparameter_change", ["R7"], "dropped lr")]
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=4)
    assert violations == []


def test_coverage_missing_decision_recorded_as_violation() -> None:
    rule_evals = [_r(0), _r(1, R7=True), _r(2), _r(3), _r(4)]
    decisions: list[dict] = []  # no one addressed R7
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=5)
    assert any(v.kind == "missing_decision" and v.rule == "R7" for v in violations)


def test_coverage_spurious_citation_flagged() -> None:
    rule_evals = [_r(0), _r(1), _r(2)]
    decisions = [_d(1, "hyperparameter_change", ["R3"], "cited rule that didn't fire")]
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=3)
    assert any(v.kind == "spurious_citation" and v.rule == "R3" for v in violations)


def test_coverage_unresolved_deferral_flagged() -> None:
    rule_evals = [_r(0, R6=True)] + [_r(i, R6=True) for i in range(1, 6)]
    decisions = [_d(0, "rule_triggered_no_action", ["R6"], "watch one epoch")]
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=6)
    # R6 never clears → deferral is unresolved
    assert any(v.kind == "unresolved_deferral" and v.rule == "R6" for v in violations)


def test_coverage_deferral_that_clears_is_ok() -> None:
    rule_evals = [_r(0, R6=True), _r(1, R6=True), _r(2), _r(3)]
    decisions = [_d(0, "rule_triggered_no_action", ["R6"], "watch one epoch")]
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=4)
    assert not any(v.kind == "unresolved_deferral" for v in violations)


def test_coverage_precedence_violation_when_r7_and_r1_fire_together() -> None:
    rule_evals = [_r(0), _r(1, R1=True, R7=True), _r(2, R7=True), _r(3), _r(4)]
    decisions = [
        _d(1, "hyperparameter_change", ["R1"], "only actioned R1"),
    ]
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=5)
    assert any(v.kind == "precedence_violation" for v in violations)


def test_coverage_precedence_ok_when_r7_is_actioned() -> None:
    rule_evals = [_r(0), _r(1, R1=True, R7=True), _r(2, R7=True), _r(3), _r(4)]
    decisions = [
        _d(1, "hyperparameter_change", ["R7"], "stabilized first"),
        _d(2, "rule_triggered_no_action", ["R1"], "deferred_to_R7"),
    ]
    violations = audit_rule_coverage(rule_evals, decisions, epochs_total=5)
    assert not any(v.kind == "precedence_violation" for v in violations)
