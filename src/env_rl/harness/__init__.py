"""Iterative Self-Refine harness for the env_rl environment.

This harness is **Iterative Self-Refine** (Madaan et al., 2023) — NOT
reinforcement learning. The distinction matters and is enforced throughout
this module:

  - Model weights NEVER change. There is no gradient, no optimizer, no policy
    update rule. The same OpenAI model is called in every attempt.
  - "Learning" is entirely in-prompt. After each attempt, the next attempt's
    system prompt is augmented with the prior attempts' scores and violations.
  - Each OpenAI conversation is independent. Nothing persists across attempts
    except what the harness explicitly injects into the prompt.

If you later want actual RL (per-decision reward, DPO/PPO on weights you
control), that is a separate feature that would live under ``src/env_rl/rl/``.
This module will never do that.
"""

#: Identifier used in logs and scripts to make the mode unambiguous.
HARNESS_MODE: str = "iterative_self_refine"

#: Rules the harness physically cannot action. The judge treats these as
#: advisory — firings do not require a matching decision and deferrals of
#: them do not need to clear.
#:
#:   R1, R5, R7 are ACTIVE — harness applies lr_new (R1/R7) and swap_activation (R5).
#:   R2  batch_size change: not supported (would require rebuilding DataLoader mid-run)
#:   R3  early stopping:    not supported (harness always runs to max_epochs)
#:   R4  add_block:         not supported (would require optimizer param-group rebuild)
#:   R6  add_bn / add_residual: not supported (can't retrofit structure safely)
HARNESS_WAIVED_RULES: frozenset[str] = frozenset({"R2", "R3", "R4", "R6"})

from env_rl.harness.policy import (
    Decision,
    DecisionPolicy,
    OpenAIDecisionPolicy,
    ScriptedDecisionPolicy,
)
from env_rl.harness.prompt import (
    DECISION_SCHEMA,
    AttemptSummary,
    build_decision_messages,
    build_iterative_system_prompt,
    playbook_text,
)

__all__ = [
    "AttemptSummary",
    "DECISION_SCHEMA",
    "Decision",
    "DecisionPolicy",
    "HARNESS_MODE",
    "HARNESS_WAIVED_RULES",
    "OpenAIDecisionPolicy",
    "ScriptedDecisionPolicy",
    "build_decision_messages",
    "build_iterative_system_prompt",
    "playbook_text",
]
