"""Top-level judge orchestrator.

Runs the 11 steps described in the design doc in strict order. The orchestrator
short-circuits on hard fails (zeroes both scores) but still surfaces which
step failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch.nn as nn

from env_rl.judge.chain import (
    read_decision_records,
    read_epoch_records,
    read_rule_eval_records,
    verify_all_logs,
)
from env_rl.judge.coverage import audit_rule_coverage
from env_rl.judge.defensibility import audit_defensibility
from env_rl.judge.deliverables import (
    HardFail,
    check_deliverables_exist,
    check_run_config,
    load_submitted_model,
)
from env_rl.judge.evaluation import evaluate_accuracy
from env_rl.judge.live_diag import run_live_diagnostic
from env_rl.judge.replay import check_architecture_matches_submission
from env_rl.judge.scoring import Scores, compute_scores


def run_judge(
    *,
    workspace: str | Path,
    judge_logs: str | Path,
    root_hash: str,
    target_acc: float,
    test_loader: Iterable,
    live_diag_batches: Iterable,
    initial_arch_spec: dict | None = None,
    defensibility_sample_size: int = 10,
    defensibility_seed: int = 42,
    live_diag_tolerance: float = 0.30,
) -> Scores:
    """Run all 11 steps. Returns the two-axis Scores object."""
    workspace = Path(workspace)
    judge_logs = Path(judge_logs)
    metrics_log = judge_logs / "metrics_log.jsonl"
    decision_log = judge_logs / "decision_log.jsonl"
    rule_log = judge_logs / "rule_evaluations.jsonl"

    try:
        # (1) deliverables
        check_deliverables_exist(workspace)
        # (2) signature + (3) load weights
        model = load_submitted_model(workspace)
        # (5) log-chain + bookends across all three logs
        session_start = verify_all_logs(judge_logs, root_hash=root_hash)
        # (4) run_config consistency
        check_run_config(workspace, session_start_record=session_start)
        # (6) architecture replay
        if initial_arch_spec is not None:
            decisions_raw = read_decision_records(decision_log)
            check_architecture_matches_submission(
                initial_arch_spec, decisions_raw, model
            )
        # (7) live diagnostic sanity
        epoch_records = read_epoch_records(metrics_log)
        if not epoch_records:
            raise HardFail("no epoch records in metrics_log.jsonl")
        run_live_diagnostic(
            model,
            live_diag_batches,
            epoch_records[-1],
            tolerance=live_diag_tolerance,
        )
    except HardFail:
        return compute_scores(
            hard_fail=True,
            test_accuracy=0.0,
            target_acc=target_acc,
            violations=0,
            total_decisions=0,
        )

    # (8) coverage audit
    rule_evals = read_rule_eval_records(rule_log)
    decisions = read_decision_records(decision_log)
    coverage_violations = audit_rule_coverage(
        rule_evals, decisions, epochs_total=len(rule_evals)
    )
    # (9) defensibility sampling
    defensibility_failures = audit_defensibility(
        decisions,
        sample_size=defensibility_sample_size,
        seed=defensibility_seed,
    )
    total_violations = len(coverage_violations) + len(defensibility_failures)

    # (10) test-set evaluation
    test_accuracy = evaluate_accuracy(model, test_loader)

    # (11) compute and return two scores
    return compute_scores(
        hard_fail=False,
        test_accuracy=test_accuracy,
        target_acc=target_acc,
        violations=total_violations,
        total_decisions=len(decisions),
    )
