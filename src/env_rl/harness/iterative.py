"""Iterative Self-Refine multi-attempt driver (NOT reinforcement learning).

Runs the training + judging pipeline N times. After each attempt, collects
the scores and violation list and feeds them into the next attempt's system
prompt. Returns the best attempt by (process_score, accuracy_score).

Explicitly NOT RL. Weights never change. See ``env_rl.harness.__init__`` for
the distinction. Every user-facing artifact this module produces (summary
files, CLI output) is labeled with ``mode = "iterative_self_refine"``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from env_rl.harness import HARNESS_MODE
from env_rl.harness.policy import OpenAIDecisionPolicy
from env_rl.harness.prompt import AttemptSummary
from env_rl.judge import run_judge
from env_rl.judge.coverage import Violation, audit_rule_coverage
from env_rl.judge.defensibility import DefensibilityFailure, audit_defensibility
from env_rl.judge.chain import (
    read_decision_records,
    read_rule_eval_records,
)
from env_rl.judge.scoring import Scores
from env_rl.monitor import session as _session_mod


@dataclass
class AttemptResult:
    index: int
    scores: Scores
    summary: AttemptSummary
    workspace: Path
    judge_logs: Path


@dataclass
class IterativeResult:
    best: AttemptResult
    all_attempts: list[AttemptResult]


def _collect_violations(judge_logs: Path) -> tuple[list[Violation], list[DefensibilityFailure]]:
    rule_evals = read_rule_eval_records(judge_logs / "rule_evaluations.jsonl")
    decisions = read_decision_records(judge_logs / "decision_log.jsonl")
    coverage = audit_rule_coverage(rule_evals, decisions, epochs_total=len(rule_evals))
    defensibility = audit_defensibility(decisions, sample_size=10, seed=42)
    return coverage, defensibility


def _violations_to_summary(
    coverage: list[Violation], defensibility: list[DefensibilityFailure]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in coverage:
        out.append({"kind": v.kind, "rule": v.rule, "epoch": v.epoch, "detail": v.detail})
    for f in defensibility:
        out.append({"kind": f.kind, "rule": f.rule, "epoch": f.epoch, "detail": f.detail})
    return out


def run_iterative(
    *,
    attempts: int,
    client: Any,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    run_one_attempt: Callable[[OpenAIDecisionPolicy, Path, Path], Scores],
    base_dir: Path,
) -> IterativeResult:
    """Run ``attempts`` full training+judge cycles; accumulate feedback.

    ``run_one_attempt(policy, workspace, judge_logs) -> Scores`` is the injected
    do-one-run function. Swapped in by the CLI (real) and tests (fake).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    prior: list[AttemptSummary] = []
    all_attempts: list[AttemptResult] = []

    for i in range(1, attempts + 1):
        # ensure a fully clean session between attempts
        _session_mod._reset_for_tests()

        # wipe any residue from a previous run — log chains are append-only
        # so even a single stale line will hard-fail the judge's bookend check
        attempt_dir = base_dir / f"attempt_{i:02d}"
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        workspace = attempt_dir / "workspace"
        judge_logs = attempt_dir / "judge_logs"
        workspace.mkdir(parents=True, exist_ok=True)
        judge_logs.mkdir(parents=True, exist_ok=True)

        policy = OpenAIDecisionPolicy(
            client=client,
            model=model_name,
            prior_attempts=prior,
            temperature=temperature,
        )

        scores = run_one_attempt(policy, workspace, judge_logs)
        coverage, defensibility = _collect_violations(judge_logs)
        violations = _violations_to_summary(coverage, defensibility)

        summary = AttemptSummary(
            attempt_index=i,
            accuracy_score=scores.accuracy_score,
            process_score=scores.process_score,
            test_accuracy=scores.test_accuracy,
            total_decisions=scores.total_decisions,
            violations=scores.violations,
            violation_summary=violations,
        )
        prior.append(summary)
        all_attempts.append(
            AttemptResult(
                index=i,
                scores=scores,
                summary=summary,
                workspace=workspace,
                judge_logs=judge_logs,
            )
        )

        (base_dir / f"attempt_{i:02d}" / "summary.json").write_text(
            json.dumps(
                {
                    "mode": HARNESS_MODE,  # iterative_self_refine — NOT reinforcement learning
                    "attempt_index": i,
                    "scores": asdict(scores),
                    "summary": {
                        "accuracy_score": summary.accuracy_score,
                        "process_score": summary.process_score,
                        "violations": summary.violations,
                        "total_decisions": summary.total_decisions,
                    },
                    "violation_summary": violations,
                },
                indent=2,
            )
        )

    best = max(
        all_attempts,
        key=lambda a: (a.scores.process_score, a.scores.accuracy_score),
    )
    return IterativeResult(best=best, all_attempts=all_attempts)
