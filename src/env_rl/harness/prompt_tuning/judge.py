"""Prompt Judge — picks the winner between the current prompt and a candidate.

The Judge looks at:
  - Pass rate on the scenario suite (primary)
  - Token cost per call (secondary — penalise bloat)
  - Regressions (scenarios the old prompt passed but the new one failed)

It returns a structured verdict with the winner and the reasoning, so the
meta-loop can persist the decision (auditable prompt evolution).
"""

from __future__ import annotations

from dataclasses import dataclass

from env_rl.harness.prompt_tuning.scenarios import ScenarioResult
from env_rl.harness.prompt_tuning.tester import pass_rate


@dataclass
class PromptJudgment:
    winner: str  # "old" | "new" | "tie"
    reason: str
    old_pass_rate: float
    new_pass_rate: float
    old_prompt_len: int
    new_prompt_len: int
    regressions: list[str]  # scenario names new failed that old passed
    improvements: list[str]  # scenario names new passed that old failed


class PromptJudge:
    """Decides which of two prompts to promote based on scenario results."""

    def __init__(self, *, length_penalty_per_char: float = 1e-5) -> None:
        # If the two prompts tie on pass rate, prefer the shorter one.
        # `length_penalty_per_char` is the equivalent "score loss" per extra
        # character, so a 4000-char prompt loses 0.04 vs a 0-char prompt.
        self._length_penalty_per_char = length_penalty_per_char

    def compare(
        self,
        *,
        old_prompt: str,
        new_prompt: str,
        old_results: list[ScenarioResult],
        new_results: list[ScenarioResult],
    ) -> PromptJudgment:
        old_pass = pass_rate(old_results)
        new_pass = pass_rate(new_results)

        old_pass_set = {r.scenario.name for r in old_results if r.passed}
        new_pass_set = {r.scenario.name for r in new_results if r.passed}
        regressions = sorted(old_pass_set - new_pass_set)
        improvements = sorted(new_pass_set - old_pass_set)

        old_score = old_pass - self._length_penalty_per_char * len(old_prompt)
        new_score = new_pass - self._length_penalty_per_char * len(new_prompt)

        if abs(new_score - old_score) < 1e-6:
            winner = "tie"
            reason = (
                f"both prompts score ~equally (old={old_score:.4f}, "
                f"new={new_score:.4f}); keeping old to avoid churn"
            )
        elif new_score > old_score:
            winner = "new"
            reason = (
                f"new prompt wins: pass_rate {old_pass:.2f} -> {new_pass:.2f} "
                f"(score {old_score:.4f} -> {new_score:.4f}); "
                f"{len(improvements)} improvements, {len(regressions)} regressions"
            )
        else:
            winner = "old"
            reason = (
                f"old prompt wins: new regressed "
                f"({old_pass:.2f} -> {new_pass:.2f}, "
                f"score {old_score:.4f} -> {new_score:.4f}); "
                f"{len(regressions)} regressions, only {len(improvements)} improvements"
            )

        return PromptJudgment(
            winner=winner,
            reason=reason,
            old_pass_rate=old_pass,
            new_pass_rate=new_pass,
            old_prompt_len=len(old_prompt),
            new_prompt_len=len(new_prompt),
            regressions=regressions,
            improvements=improvements,
        )
