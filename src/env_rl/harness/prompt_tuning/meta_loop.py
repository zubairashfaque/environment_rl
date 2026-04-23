"""Meta-loop: Tuner → Tester → Judge across multiple attempts.

After each attempt the meta-loop:
  1. Collects violations from the attempt's coverage + defensibility audits
  2. Asks the PromptTuner to propose an edit to the current champion prompt
  3. Runs the PromptTester on the scenario suite for both old and new prompts
  4. Hands both to the PromptJudge; promotes the winner
  5. Persists every prompt version and the full verdict history to disk

The champion prompt becomes the starting point for the next attempt's system
prompt (prior-attempt feedback still gets appended on top by the existing
``build_iterative_system_prompt`` flow).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from env_rl.harness.prompt_tuning.judge import PromptJudge, PromptJudgment
from env_rl.harness.prompt_tuning.scenarios import ScenarioResult
from env_rl.harness.prompt_tuning.tester import PromptTester, pass_rate
from env_rl.harness.prompt_tuning.tuner import PromptEdit, PromptTuner


@dataclass
class PromptVersion:
    version: int
    parent: int | None
    technique: str
    rationale: str
    path: str
    scenario_pass_rate: float
    prompt_len: int


@dataclass
class MetaLoopIteration:
    attempt_index: int
    proposed_edit_technique: str
    proposed_edit_rationale: str
    old_version: int
    new_version: int
    old_pass_rate: float
    new_pass_rate: float
    winner: str  # "old" | "new" | "tie"
    judge_reason: str
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)


class MetaLoop:
    """Orchestrates Tuner → Tester → Judge across attempts."""

    def __init__(
        self,
        *,
        base_dir: Path,
        initial_prompt: str,
        tester_client: Any,
        tester_model: str = "gpt-4o-mini",
        tuner: PromptTuner | None = None,
        judge: PromptJudge | None = None,
        scoreboard: Any = None,
        tracer: Any = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._prompts_dir = self._base_dir / "prompts"
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        self._tuner = tuner or PromptTuner()
        self._tester = PromptTester(client=tester_client, model=tester_model)
        self._judge = judge or PromptJudge()
        if scoreboard is None:
            from env_rl.harness.prompt_tuning.scoreboard import TechniqueScoreboard
            scoreboard = TechniqueScoreboard(
                path=self._base_dir / ".scoreboard.json"
            )
        self._scoreboard = scoreboard

        # Shared agent tracer (None → no-op)
        from env_rl.harness.agent_trace import NULL_TRACER
        self._tracer = tracer or NULL_TRACER

        self._versions: list[PromptVersion] = []
        self._iterations: list[MetaLoopIteration] = []

        # Evaluate the initial prompt once so we have a baseline pass rate
        initial_results = self._tester.run_suite(initial_prompt)
        v0 = self._persist(
            version=0, parent=None, technique="initial",
            rationale="starting prompt (playbook + response rules)",
            prompt=initial_prompt,
            scenario_pass_rate=pass_rate(initial_results),
        )
        self._versions.append(v0)
        self._champion_prompt = initial_prompt
        self._champion_version = 0
        self._champion_results = initial_results

    # -- public API ----------------------------------------------------------

    @property
    def champion_prompt(self) -> str:
        return self._champion_prompt

    @property
    def champion_version(self) -> int:
        return self._champion_version

    def step(
        self,
        *,
        attempt_index: int,
        violations: list[dict],
    ) -> MetaLoopIteration:
        """One round of the meta-loop.

        Returns the ``MetaLoopIteration`` describing what changed (if
        anything). Promotes the winning prompt into ``self.champion_prompt``.
        """
        # --- Tuner -----------------------------------------------------
        with self._tracer.timed(
            agent="tuner", action="propose_edit",
            input_summary={"attempt_index": attempt_index,
                           "violation_count": len(violations)},
        ) as trace_out:
            edit: PromptEdit = self._tuner.propose_edit(
                violations=violations, attempt_index=attempt_index,
            )
            trace_out["output_summary"] = {
                "technique": edit.technique,
                "rationale": edit.rationale,
                "addition_len": len(edit.addition),
            }

        candidate = edit.apply(self._champion_prompt)

        # No real edit proposed -> skip tester round
        if candidate == self._champion_prompt:
            return self._record_noop(attempt_index=attempt_index, edit=edit)

        # --- Tester ----------------------------------------------------
        with self._tracer.timed(
            agent="tester", action="run_suite",
            input_summary={"prompt_version_parent": self._champion_version,
                           "prompt_len": len(candidate)},
        ) as trace_out:
            new_results = self._tester.run_suite(candidate)
            trace_out["output_summary"] = {
                "pass_rate": pass_rate(new_results),
                "passed": sum(1 for r in new_results if r.passed),
                "total": len(new_results),
            }

        # --- Judge -----------------------------------------------------
        with self._tracer.timed(
            agent="judge", action="compare_prompts",
            input_summary={"old_version": self._champion_version,
                           "old_len": len(self._champion_prompt),
                           "new_len": len(candidate)},
        ) as trace_out:
            verdict: PromptJudgment = self._judge.compare(
                old_prompt=self._champion_prompt,
                new_prompt=candidate,
                old_results=self._champion_results,
                new_results=new_results,
            )
            trace_out["output_summary"] = {
                "winner": verdict.winner,
                "old_pass_rate": verdict.old_pass_rate,
                "new_pass_rate": verdict.new_pass_rate,
                "improvements": verdict.improvements,
                "regressions": verdict.regressions,
            }

        new_version = self._versions[-1].version + 1
        v = self._persist(
            version=new_version,
            parent=self._champion_version,
            technique=edit.technique,
            rationale=edit.rationale,
            prompt=candidate,
            scenario_pass_rate=verdict.new_pass_rate,
        )
        self._versions.append(v)

        if verdict.winner == "new":
            self._champion_prompt = candidate
            self._champion_version = new_version
            self._champion_results = new_results

        # Record technique outcome in the scoreboard
        outcome = (
            "win" if verdict.winner == "new"
            else ("loss" if verdict.winner == "old" else "tie")
        )
        try:
            self._scoreboard.record(edit.technique, outcome)
            self._tracer.record(
                agent="scoreboard", action="record",
                input_summary={"technique": edit.technique},
                output_summary={"outcome": outcome},
            )
        except Exception:  # noqa: BLE001
            pass  # scoreboard is informational; never let it crash the run

        it = MetaLoopIteration(
            attempt_index=attempt_index,
            proposed_edit_technique=edit.technique,
            proposed_edit_rationale=edit.rationale,
            old_version=self._versions[-2].version if len(self._versions) >= 2 else 0,
            new_version=new_version,
            old_pass_rate=verdict.old_pass_rate,
            new_pass_rate=verdict.new_pass_rate,
            winner=verdict.winner,
            judge_reason=verdict.reason,
            improvements=list(verdict.improvements),
            regressions=list(verdict.regressions),
        )
        self._iterations.append(it)
        self._persist_log()
        return it

    # -- helpers ------------------------------------------------------------

    def _record_noop(
        self, *, attempt_index: int, edit: PromptEdit
    ) -> MetaLoopIteration:
        it = MetaLoopIteration(
            attempt_index=attempt_index,
            proposed_edit_technique=edit.technique,
            proposed_edit_rationale=edit.rationale or "no violations to address",
            old_version=self._champion_version,
            new_version=self._champion_version,
            old_pass_rate=pass_rate(self._champion_results),
            new_pass_rate=pass_rate(self._champion_results),
            winner="tie",
            judge_reason="tuner proposed no edit; champion retained",
            improvements=[],
            regressions=[],
        )
        self._iterations.append(it)
        self._persist_log()
        return it

    def _persist(
        self, *,
        version: int,
        parent: int | None,
        technique: str,
        rationale: str,
        prompt: str,
        scenario_pass_rate: float,
    ) -> PromptVersion:
        path = self._prompts_dir / f"v{version:03d}.txt"
        path.write_text(prompt)
        return PromptVersion(
            version=version,
            parent=parent,
            technique=technique,
            rationale=rationale,
            path=str(path),
            scenario_pass_rate=scenario_pass_rate,
            prompt_len=len(prompt),
        )

    def _persist_log(self) -> None:
        log = {
            "champion_version": self._champion_version,
            "versions": [asdict(v) for v in self._versions],
            "iterations": [asdict(i) for i in self._iterations],
            "scoreboard": self._scoreboard.summary(),
        }
        (self._base_dir / "meta_loop_log.json").write_text(
            json.dumps(log, indent=2)
        )

    @property
    def scoreboard_summary(self) -> dict:
        return self._scoreboard.summary()
