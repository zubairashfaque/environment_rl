from pathlib import Path

from env_rl.harness.iterative import IterativeResult, run_iterative
from env_rl.judge.scoring import Scores
from tests.unit.harness._stubs import StubOpenAIClient


def _stub_scores(acc: float, proc: float, viols: int, decisions: int) -> Scores:
    return Scores(
        accuracy_score=acc,
        process_score=proc,
        hard_fail=False,
        test_accuracy=acc,
        violations=viols,
        total_decisions=decisions,
    )


def test_iterative_runs_n_attempts_and_returns_best(tmp_path: Path) -> None:
    # Fake "attempts get better each iteration"
    fake_score_sequence = [
        _stub_scores(0.2, 0.5, 2, 4),
        _stub_scores(0.3, 0.8, 1, 5),
        _stub_scores(0.5, 1.0, 0, 5),
    ]
    calls: list[int] = []

    def fake_run_one(policy, workspace, judge_logs):
        # write empty log files so violation collector doesn't raise
        (judge_logs / "rule_evaluations.jsonl").write_text("")
        (judge_logs / "decision_log.jsonl").write_text("")
        idx = len(calls)
        calls.append(idx)
        return fake_score_sequence[idx]

    # OpenAI stub — never actually called by the fake runner
    client = StubOpenAIClient(responses=[])

    result = run_iterative(
        attempts=3,
        client=client,
        run_one_attempt=fake_run_one,
        base_dir=tmp_path,
    )
    assert isinstance(result, IterativeResult)
    assert len(result.all_attempts) == 3
    # best is by (process_score, accuracy_score) — attempt 3
    assert result.best.index == 3
    assert result.best.scores.process_score == 1.0
    assert result.best.scores.accuracy_score == 0.5


def test_iterative_accumulates_prior_feedback_into_later_attempts(
    tmp_path: Path,
) -> None:
    # Log the system prompt each attempt saw, to verify feedback carries forward.
    observed_prompts: list[str] = []

    def fake_run_one(policy, workspace, judge_logs):
        observed_prompts.append(policy.system_prompt)
        (judge_logs / "rule_evaluations.jsonl").write_text("")
        (judge_logs / "decision_log.jsonl").write_text("")
        return _stub_scores(0.3, 0.7, 1, 4)

    client = StubOpenAIClient(responses=[])
    run_iterative(
        attempts=3,
        client=client,
        run_one_attempt=fake_run_one,
        base_dir=tmp_path,
    )

    assert len(observed_prompts) == 3
    assert "PRIOR ATTEMPTS" not in observed_prompts[0]
    assert "PRIOR ATTEMPTS" in observed_prompts[1]
    assert "Attempt 1" in observed_prompts[1]
    # Third attempt sees two priors
    assert "Attempt 2" in observed_prompts[2]
