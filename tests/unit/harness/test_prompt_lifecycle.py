"""Tests for prompt-tuning lifecycle controls: default-on, reset, resume."""

import json
from pathlib import Path

import pytest

from env_rl.harness.prompt_tuning.meta_loop import MetaLoop
from env_rl.harness.prompt_tuning.scenarios import SCENARIO_SUITE
from tests.unit.harness._stubs import StubOpenAIClient


def _good_response(rule: str = "R7") -> dict:
    return {
        "event_type": "hyperparameter_change",
        "cites": [rule],
        "justification": "drop lr",
        "remedy_direction": "decrease_lr",
        "remedy_params": {"lr_new": 0.03, "edit_op": "none", "edit_to": "none"},
    }


def test_meta_loop_default_initial_prompt(tmp_path: Path) -> None:
    """Without a seed path, MetaLoop uses the supplied initial prompt."""
    n = len(SCENARIO_SUITE)
    client = StubOpenAIClient(responses=[_good_response()] * n)
    ml = MetaLoop(
        base_dir=tmp_path,
        initial_prompt="DEFAULT_PLAYBOOK_PROMPT",
        tester_client=client,
    )
    assert ml.champion_prompt == "DEFAULT_PLAYBOOK_PROMPT"
    # v000.txt contains the initial prompt
    assert (tmp_path / "prompts" / "v000.txt").read_text() == "DEFAULT_PLAYBOOK_PROMPT"


def test_meta_loop_seed_prompt_path_overrides_initial(tmp_path: Path) -> None:
    """With a seed path, MetaLoop resumes from that file."""
    seed = tmp_path / "saved_champion.txt"
    seed.write_text("EVOLVED_PROMPT_FROM_PRIOR_RUN")

    n = len(SCENARIO_SUITE)
    client = StubOpenAIClient(responses=[_good_response()] * n)
    ml = MetaLoop(
        base_dir=tmp_path,
        initial_prompt="SHOULD_BE_IGNORED",
        tester_client=client,
        seed_prompt_path=seed,
    )
    assert ml.champion_prompt == "EVOLVED_PROMPT_FROM_PRIOR_RUN"
    # v000.txt now reflects the resumed prompt
    assert (tmp_path / "prompts" / "v000.txt").read_text() == "EVOLVED_PROMPT_FROM_PRIOR_RUN"
    # The in-memory version record reflects the resume path
    # (meta_loop_log.json is only written after step(), not on init)
    assert ml._versions[0].technique == "resumed_champion"  # noqa: SLF001


def test_meta_loop_seed_path_missing_falls_back_to_initial(tmp_path: Path) -> None:
    """If the seed file does not exist, fall through to initial_prompt."""
    n = len(SCENARIO_SUITE)
    client = StubOpenAIClient(responses=[_good_response()] * n)
    ml = MetaLoop(
        base_dir=tmp_path,
        initial_prompt="FALLBACK_PROMPT",
        tester_client=client,
        seed_prompt_path=tmp_path / "nope.txt",
    )
    assert ml.champion_prompt == "FALLBACK_PROMPT"


def test_run_iterative_meta_loop_default_is_true() -> None:
    """The default value of meta_loop should be True (prompt tuning on)."""
    import inspect
    from env_rl.harness.iterative import run_iterative

    sig = inspect.signature(run_iterative)
    assert sig.parameters["meta_loop"].default is True


def test_run_iterative_reset_prompt_history_wipes_expected_paths(
    tmp_path: Path,
) -> None:
    """Calling run_iterative with reset_prompt_history=True deletes the
    three lifecycle artifacts but leaves attempt dirs alone."""
    from env_rl.judge.scoring import Scores
    from env_rl.harness.iterative import run_iterative

    # Seed a base_dir with the three artifacts + a fake attempt_01
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "v000.txt").write_text("old")
    (tmp_path / ".scoreboard.json").write_text("{}")
    (tmp_path / "meta_loop_log.json").write_text("{}")
    (tmp_path / "attempt_01").mkdir()
    (tmp_path / "attempt_01" / "keep.txt").write_text("keep me")

    def fake_run(policy, workspace, judge_logs):
        (judge_logs / "rule_evaluations.jsonl").write_text("")
        (judge_logs / "decision_log.jsonl").write_text("")
        return Scores(
            accuracy_score=0.5, process_score=0.5, hard_fail=False,
            test_accuracy=0.5, violations=0, total_decisions=1,
        )

    # meta_loop=False so MetaLoop doesn't try to init — just exercise reset
    run_iterative(
        attempts=1,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
        reset_prompt_history=True,
    )

    # The three lifecycle artifacts should have been deleted
    assert not (tmp_path / "prompts").exists()
    assert not (tmp_path / ".scoreboard.json").exists()
    assert not (tmp_path / "meta_loop_log.json").exists()
    # attempt_01 is deleted by run_iterative's normal per-attempt cleanup
    # (the attempt_NN dir always gets wiped before each attempt). To confirm
    # reset_prompt_history does not break that path, we just check the new
    # attempt dir was created by the run.
    assert (tmp_path / "attempt_01").exists()


def test_run_iterative_no_meta_loop_prevents_prompts_dir_creation(
    tmp_path: Path,
) -> None:
    """When meta_loop=False, no prompts/ directory should be created."""
    from env_rl.judge.scoring import Scores
    from env_rl.harness.iterative import run_iterative

    def fake_run(policy, workspace, judge_logs):
        (judge_logs / "rule_evaluations.jsonl").write_text("")
        (judge_logs / "decision_log.jsonl").write_text("")
        return Scores(
            accuracy_score=0.5, process_score=0.5, hard_fail=False,
            test_accuracy=0.5, violations=0, total_decisions=1,
        )

    run_iterative(
        attempts=1,
        client=StubOpenAIClient(responses=[]),
        run_one_attempt=fake_run,
        base_dir=tmp_path,
        meta_loop=False,
    )
    assert not (tmp_path / "prompts").exists()
    assert not (tmp_path / "meta_loop_log.json").exists()
