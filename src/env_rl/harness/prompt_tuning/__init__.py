"""Multi-agent prompt evolution for the iterative self-refine harness.

Three agents work together to improve the decision-LLM's system prompt
across iterations without ever touching model weights:

  - PromptTuner      — proposes edits to the prompt based on violations
  - PromptTester     — stress-tests candidate prompts on synthetic scenarios
  - PromptJudge      — picks the winner between old and new prompt

The meta-loop orchestrates these: Tuner proposes → Tester evaluates both
old and new → Judge promotes the better one. Every prompt version is
versioned and persisted under ``prompts/`` so any attempt is reproducible.
"""

from env_rl.harness.prompt_tuning.scenarios import (
    SCENARIO_SUITE,
    Scenario,
    ScenarioResult,
)
from env_rl.harness.prompt_tuning.tester import PromptTester
from env_rl.harness.prompt_tuning.tuner import (
    PROMPT_TECHNIQUES,
    PromptEdit,
    PromptTuner,
)
from env_rl.harness.prompt_tuning.judge import PromptJudge, PromptJudgment

__all__ = [
    "PROMPT_TECHNIQUES",
    "PromptEdit",
    "PromptJudge",
    "PromptJudgment",
    "PromptTester",
    "PromptTuner",
    "SCENARIO_SUITE",
    "Scenario",
    "ScenarioResult",
]
