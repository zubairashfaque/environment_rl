"""Tests for restart-class edits (add_block/remove_block/add_bn).

When the LLM decides a restart-class architecture edit, the harness must:
  1. NOT mutate the live model mid-run.
  2. Log the decision as rule_triggered_no_action with a "restart scheduled"
     justification.
  3. End the current attempt cleanly at the current epoch.
  4. Pass cfg_overrides to the NEXT attempt so it starts from epoch 0 with
     the updated architecture baked into run_config.json.

These tests exercise the iterative.py ↔ callback contract; the end-to-end
training+judging path is covered by run_llm_agent.py's synthetic smoke run.
"""

from pathlib import Path

from env_rl.agent.reference_run import PendingRestart
from env_rl.harness.iterative import run_iterative
from env_rl.judge.scoring import Scores
from tests.unit.harness._stubs import StubOpenAIClient


def _empty_logs(judge_logs: Path) -> None:
    (judge_logs / "rule_evaluations.jsonl").write_text("")
    (judge_logs / "decision_log.jsonl").write_text("")


def _scores(acc: float = 0.4, proc: float = 0.7) -> Scores:
    return Scores(
        accuracy_score=acc, process_score=proc, hard_fail=False,
        test_accuracy=acc, violations=1, total_decisions=4,
    )


def test_add_block_pending_restart_advances_num_blocks_next_attempt(
    tmp_path: Path,
) -> None:
    """Attempt 1 schedules add_block → attempt 2 sees num_blocks=3 via cfg_overrides."""
    observed_overrides: list[dict] = []

    def fake_run(policy, workspace, judge_logs, *, cfg_overrides=None):
        _empty_logs(judge_logs)
        observed_overrides.append(dict(cfg_overrides or {}))
        if len(observed_overrides) == 1:
            # Attempt 1 triggers an add_block restart at epoch 5
            return _scores(), PendingRestart(
                reason="add_block",
                triggered_at_epoch=5,
                cited_rule="R4",
                num_blocks_delta=1,
                activation="relu",
                preserved_num_blocks=2,
            )
        # Attempt 2: no restart
        return _scores(), None

    run_iterative(
        attempts=2,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
    )

    # Attempt 1 saw a fresh (empty) cfg_overrides
    assert observed_overrides[0] == {}
    # Attempt 2's overrides include num_blocks=3 (2 + 1)
    assert observed_overrides[1]["num_blocks"] == 3
    # And the activation carries forward unchanged
    assert observed_overrides[1]["activation"] == "relu"


def test_activation_swap_does_not_trigger_restart(tmp_path: Path) -> None:
    """swap_activation is CONTINUE class; next attempt sees no cfg_overrides."""
    observed_overrides: list[dict] = []

    def fake_run(policy, workspace, judge_logs, *, cfg_overrides=None):
        _empty_logs(judge_logs)
        observed_overrides.append(dict(cfg_overrides or {}))
        # Neither attempt schedules a restart
        return _scores(), None

    run_iterative(
        attempts=2,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
    )

    assert observed_overrides[0] == {}
    assert observed_overrides[1] == {}


def test_activation_carries_forward_through_restart(tmp_path: Path) -> None:
    """If a prior swap changed activation to leaky_relu, a later add_block
    restart preserves leaky_relu in the next attempt's cfg."""
    observed_overrides: list[dict] = []

    def fake_run(policy, workspace, judge_logs, *, cfg_overrides=None):
        _empty_logs(judge_logs)
        observed_overrides.append(dict(cfg_overrides or {}))
        if len(observed_overrides) == 1:
            return _scores(), PendingRestart(
                reason="add_block",
                triggered_at_epoch=3,
                cited_rule="R4",
                num_blocks_delta=1,
                activation="leaky_relu",  # prior swap
                preserved_num_blocks=2,
            )
        return _scores(), None

    run_iterative(
        attempts=2,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
    )

    assert observed_overrides[1]["activation"] == "leaky_relu"
    assert observed_overrides[1]["num_blocks"] == 3


def test_attempts_cap_honored_even_when_every_attempt_restarts(
    tmp_path: Path,
) -> None:
    """--attempts caps the total runs even if every attempt schedules a restart."""
    call_count = 0

    def fake_run(policy, workspace, judge_logs, *, cfg_overrides=None):
        nonlocal call_count
        call_count += 1
        _empty_logs(judge_logs)
        return _scores(), PendingRestart(
            reason="add_block",
            triggered_at_epoch=2,
            cited_rule="R4",
            num_blocks_delta=1,
            activation="relu",
            preserved_num_blocks=2 + (call_count - 1),  # grows each time
        )

    result = run_iterative(
        attempts=3,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
    )

    # Exactly 3 attempts ran, no more
    assert call_count == 3
    assert len(result.all_attempts) == 3


def test_pending_restart_written_to_summary_json(tmp_path: Path) -> None:
    """summary.json records pending_restart metadata for the audit trail."""
    import json

    def fake_run(policy, workspace, judge_logs, *, cfg_overrides=None):
        _empty_logs(judge_logs)
        return _scores(), PendingRestart(
            reason="add_block",
            triggered_at_epoch=4,
            cited_rule="R4",
            num_blocks_delta=1,
            activation="relu",
            preserved_num_blocks=2,
        )

    run_iterative(
        attempts=1,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
    )

    summary = json.loads(
        (tmp_path / "attempt_01" / "summary.json").read_text()
    )
    pr = summary["pending_restart"]
    assert pr is not None
    assert pr["reason"] == "add_block"
    assert pr["cited_rule"] == "R4"
    assert pr["num_blocks_delta"] == 1
    assert pr["triggered_at_epoch"] == 4
