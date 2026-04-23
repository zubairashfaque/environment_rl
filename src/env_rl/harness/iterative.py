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
from env_rl.harness.prompt import (
    AttemptSummary,
    build_iterative_system_prompt,
)


def _build_initial_prompt(prior_attempts: list) -> str:
    return build_iterative_system_prompt(prior_attempts=prior_attempts)
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


def _collect_violations(
    judge_logs: Path,
    waived_rules: frozenset[str] | set[str] | None = None,
) -> tuple[list[Violation], list[DefensibilityFailure]]:
    rule_evals = read_rule_eval_records(judge_logs / "rule_evaluations.jsonl")
    decisions = read_decision_records(judge_logs / "decision_log.jsonl")
    coverage = audit_rule_coverage(
        rule_evals, decisions,
        epochs_total=len(rule_evals),
        waived_rules=waived_rules,
    )
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
    waived_rules: frozenset[str] | set[str] | None = None,
    meta_loop: bool = True,
    reset_prompt_history: bool = False,
    resume_from_champion: bool = False,
) -> IterativeResult:
    """Run ``attempts`` full training+judge cycles; accumulate feedback.

    ``run_one_attempt(policy, workspace, judge_logs) -> Scores`` is the injected
    do-one-run function. Swapped in by the CLI (real) and tests (fake).

    ``meta_loop`` is ON by default: after each attempt a Tuner→Tester→Judge
    cycle proposes a prompt edit, evaluates it on the scenario suite, and
    promotes the winner — so the prompt evolves across attempts. Set to
    False to run with a static (feedback-only) prompt.

    ``reset_prompt_history=True`` wipes ``base_dir/prompts/``,
    ``.scoreboard.json``, and ``meta_loop_log.json`` at the start of the run.

    ``resume_from_champion=True`` loads the highest-numbered
    ``base_dir/prompts/v*.txt`` from a prior run as the MetaLoop's v000.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # --- prompt-tuning lifecycle controls ---------------------------------
    if reset_prompt_history:
        prompts_dir = base_dir / "prompts"
        if prompts_dir.exists():
            shutil.rmtree(prompts_dir)
        for name in (".scoreboard.json", "meta_loop_log.json"):
            p = base_dir / name
            if p.exists():
                p.unlink()
        print(f"[prompt-tuning] reset history at {base_dir}")

    seed_prompt_path = None
    if resume_from_champion and meta_loop:
        prompts_dir = base_dir / "prompts"
        if prompts_dir.exists():
            candidates = sorted(prompts_dir.glob("v*.txt"))
            if candidates:
                seed_prompt_path = candidates[-1]
                print(f"[prompt-tuning] resuming from {seed_prompt_path}")

    # Startup banner
    if meta_loop:
        print(
            "[prompt-tuning] ON  "
            f"(base_dir={base_dir}, --no-meta-loop to disable)"
        )
    else:
        print("[prompt-tuning] OFF  (static feedback-only prompt)")

    prior: list[AttemptSummary] = []
    all_attempts: list[AttemptResult] = []

    # Optional meta-loop (Tuner/Tester/Judge) — initialised lazily with the
    # initial (prior-free) system prompt on the first attempt.
    ml = None  # type: ignore[assignment]

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

        # Persist the iterative-feedback block that will be fed into THIS
        # attempt's system prompt, so it's easy to see the self-refine loop
        # at work without having to diff transcripts.
        feedback_record = {
            "mode": HARNESS_MODE,
            "attempt_index": i,
            "prior_attempts": [
                {
                    "attempt_index": p.attempt_index,
                    "accuracy_score": p.accuracy_score,
                    "process_score": p.process_score,
                    "test_accuracy": p.test_accuracy,
                    "total_decisions": p.total_decisions,
                    "violations": p.violations,
                    "violation_summary": p.violation_summary,
                }
                for p in prior
            ],
            "system_prompt_excerpt": (
                "" if not prior
                else build_iterative_system_prompt(
                    prior, include_playbook=False
                )
            ),
        }
        (attempt_dir / "feedback_in.json").write_text(
            json.dumps(feedback_record, indent=2)
        )

        # Shared per-attempt agent tracer
        from env_rl.harness.agent_trace import AgentTracer
        tracer = AgentTracer(attempt_dir / "agent_trace.jsonl")
        tracer.set_attempt_index(i)

        # If meta-loop is active, the champion prompt replaces the normal
        # system prompt (prior-attempt feedback still gets appended on top
        # via the usual OpenAIDecisionPolicy constructor — except when
        # meta_loop is on, the champion is itself already an evolved prompt
        # so we pass it directly).
        if meta_loop:
            from env_rl.harness.prompt_tuning.meta_loop import MetaLoop
            if ml is None:
                base_initial = _build_initial_prompt(prior_attempts=[])
                ml = MetaLoop(
                    base_dir=base_dir,
                    initial_prompt=base_initial,
                    tester_client=client,
                    tester_model=model_name,
                    tracer=tracer,
                    seed_prompt_path=seed_prompt_path,
                )
            else:
                # On subsequent attempts, update the existing meta-loop's
                # tracer so events land in the right attempt_NN folder.
                ml._tracer = tracer  # noqa: SLF001
            system_prompt_override = ml.champion_prompt
        else:
            system_prompt_override = None

        policy = OpenAIDecisionPolicy(
            client=client,
            model=model_name,
            prior_attempts=prior,
            temperature=temperature,
            transcript_path=attempt_dir / "llm_transcript.jsonl",
            system_prompt=system_prompt_override,
            tracer=tracer,
        )

        scores = run_one_attempt(policy, workspace, judge_logs)
        coverage, defensibility = _collect_violations(judge_logs, waived_rules=waived_rules)
        violations = _violations_to_summary(coverage, defensibility)

        # Let the meta-loop propose and evaluate an edit based on this
        # attempt's violations. The promoted champion becomes the starting
        # point for the next attempt.
        if meta_loop and ml is not None:
            ml.step(attempt_index=i, violations=violations)

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
