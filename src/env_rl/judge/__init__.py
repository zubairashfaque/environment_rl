"""Top-level judge orchestrator.

Runs the 11 steps described in the design doc in strict order. The orchestrator
short-circuits on hard fails (zeroes both scores) but still surfaces which
step failed.

If ``judge_trace_path`` is provided, writes a per-step pass/fail/detail trace
as JSON so you can see exactly which step hard-failed and why.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
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


@dataclass
class StepRecord:
    step: int
    name: str
    kind: str  # "hard_fail_gate" | "process_only" | "scoring"
    status: str  # "pass" | "hard_fail" | "skipped"
    duration_ms: float
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _write_trace(path: Path | None, steps: list[StepRecord], scores: Any) -> None:
    if path is None:
        return
    out = {
        "steps": [asdict(s) for s in steps],
        "final_scores": asdict(scores) if scores is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


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
    waived_rules: frozenset[str] | set[str] | None = None,
    judge_trace_path: str | Path | None = None,
) -> Scores:
    """Run all 11 steps. Returns the two-axis Scores object.

    When ``judge_trace_path`` is given, writes a detailed per-step trace to
    that path as JSON (step number, name, pass/fail, duration, detail).
    """
    workspace = Path(workspace)
    judge_logs = Path(judge_logs)
    trace_path = Path(judge_trace_path) if judge_trace_path else None
    metrics_log = judge_logs / "metrics_log.jsonl"
    decision_log = judge_logs / "decision_log.jsonl"
    rule_log = judge_logs / "rule_evaluations.jsonl"

    steps: list[StepRecord] = []

    def _run_step(
        num: int, name: str, kind: str, fn: Any
    ) -> tuple[bool, Any]:
        t = time.perf_counter()
        try:
            result = fn()
            steps.append(
                StepRecord(num, name, kind, "pass", (time.perf_counter() - t) * 1000)
            )
            return True, result
        except HardFail as e:
            steps.append(
                StepRecord(
                    num, name, kind, "hard_fail",
                    (time.perf_counter() - t) * 1000,
                    detail=str(e),
                )
            )
            return False, None

    # (1) deliverables
    ok, _ = _run_step(1, "deliverables_exist", "hard_fail_gate",
                       lambda: check_deliverables_exist(workspace))
    if not ok:
        scores = compute_scores(hard_fail=True, test_accuracy=0.0, target_acc=target_acc,
                                violations=0, total_decisions=0)
        _write_trace(trace_path, steps, scores)
        return scores

    # (2) + (3) signature + load model
    ok, model = _run_step(2, "signature_and_load", "hard_fail_gate",
                           lambda: load_submitted_model(workspace))
    if not ok:
        scores = compute_scores(hard_fail=True, test_accuracy=0.0, target_acc=target_acc,
                                violations=0, total_decisions=0)
        _write_trace(trace_path, steps, scores)
        return scores

    # (5) chain + bookends (run before step 4 so we have session_start)
    ok, session_start = _run_step(5, "chain_integrity", "hard_fail_gate",
                                   lambda: verify_all_logs(judge_logs, root_hash=root_hash))
    if not ok:
        scores = compute_scores(hard_fail=True, test_accuracy=0.0, target_acc=target_acc,
                                violations=0, total_decisions=0)
        _write_trace(trace_path, steps, scores)
        return scores

    # (4) run_config consistency
    ok, _ = _run_step(4, "run_config_consistency", "hard_fail_gate",
                       lambda: check_run_config(workspace, session_start_record=session_start))
    if not ok:
        scores = compute_scores(hard_fail=True, test_accuracy=0.0, target_acc=target_acc,
                                violations=0, total_decisions=0)
        _write_trace(trace_path, steps, scores)
        return scores

    # (6) architecture replay
    if initial_arch_spec is not None:
        def _replay():
            decisions_raw = read_decision_records(decision_log)
            return check_architecture_matches_submission(
                initial_arch_spec, decisions_raw, model
            )
        ok, _ = _run_step(6, "architecture_replay", "hard_fail_gate", _replay)
        if not ok:
            scores = compute_scores(hard_fail=True, test_accuracy=0.0, target_acc=target_acc,
                                    violations=0, total_decisions=0)
            _write_trace(trace_path, steps, scores)
            return scores
    else:
        steps.append(StepRecord(6, "architecture_replay", "hard_fail_gate",
                                "skipped", 0.0, "no initial_arch_spec provided"))

    # (7) live diagnostic sanity
    def _live_diag():
        epoch_records = read_epoch_records(metrics_log)
        if not epoch_records:
            raise HardFail("no epoch records in metrics_log.jsonl")
        live = run_live_diagnostic(
            model, live_diag_batches, epoch_records[-1],
            tolerance=live_diag_tolerance,
        )
        return live
    ok, live_norms = _run_step(7, "live_diagnostic_sanity", "hard_fail_gate", _live_diag)
    if not ok:
        scores = compute_scores(hard_fail=True, test_accuracy=0.0, target_acc=target_acc,
                                violations=0, total_decisions=0)
        _write_trace(trace_path, steps, scores)
        return scores
    steps[-1].extra["live_max"] = max(live_norms.values()) if live_norms else 0.0

    # (8) coverage audit
    t = time.perf_counter()
    rule_evals = read_rule_eval_records(rule_log)
    decisions = read_decision_records(decision_log)
    coverage_violations = audit_rule_coverage(
        rule_evals, decisions,
        epochs_total=len(rule_evals),
        waived_rules=waived_rules,
    )
    steps.append(StepRecord(
        8, "rule_coverage_audit", "process_only", "pass",
        (time.perf_counter() - t) * 1000,
        detail=f"{len(coverage_violations)} coverage violations",
        extra={"violation_count": len(coverage_violations),
               "violation_kinds": sorted({v.kind for v in coverage_violations})},
    ))

    # (9) defensibility sampling
    t = time.perf_counter()
    defensibility_failures = audit_defensibility(
        decisions,
        sample_size=defensibility_sample_size,
        seed=defensibility_seed,
    )
    steps.append(StepRecord(
        9, "decision_defensibility", "process_only", "pass",
        (time.perf_counter() - t) * 1000,
        detail=f"{len(defensibility_failures)} defensibility failures",
        extra={"failure_count": len(defensibility_failures),
               "failure_kinds": sorted({f.kind for f in defensibility_failures})},
    ))

    total_violations = len(coverage_violations) + len(defensibility_failures)

    # (10) test-set evaluation
    t = time.perf_counter()
    test_accuracy = evaluate_accuracy(model, test_loader)
    steps.append(StepRecord(
        10, "test_set_evaluation", "scoring", "pass",
        (time.perf_counter() - t) * 1000,
        detail=f"test_accuracy = {test_accuracy:.4f}",
        extra={"test_accuracy": test_accuracy},
    ))

    # (11) compute and return two scores
    t = time.perf_counter()
    scores = compute_scores(
        hard_fail=False,
        test_accuracy=test_accuracy,
        target_acc=target_acc,
        violations=total_violations,
        total_decisions=len(decisions),
    )
    steps.append(StepRecord(
        11, "emit_two_scores", "scoring", "pass",
        (time.perf_counter() - t) * 1000,
        detail=f"accuracy={scores.accuracy_score:.3f} process={scores.process_score:.3f}",
        extra={"accuracy_score": scores.accuracy_score,
               "process_score": scores.process_score,
               "violations": total_violations,
               "total_decisions": len(decisions)},
    ))

    _write_trace(trace_path, steps, scores)
    return scores
