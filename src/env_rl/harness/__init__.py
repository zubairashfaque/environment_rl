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

#: Rules the harness physically cannot action (shrinks over time as more
#: remedies get implemented). The judge treats these as advisory — firings
#: do not require a matching decision and deferrals of them do not need to
#: clear.
#:
#:   ACTIVE (must be actioned correctly):
#:     R1  learning rate  → harness applies lr_new to every param group
#:     R3  early stopping → remedy_direction="stop" exits the epoch loop
#:     R4  add_block      → harness appends a ResBlock and registers params
#:     R5  swap_activation → harness swaps ReLU/LeakyReLU/GELU in place
#:     R7  exploding       → same code path as R1 (LR drop)
#:
#:   WAIVED (still — pending implementation):
#:     R2  batch_size change     → DataLoader rebuild not wired up yet
#:     R6  add_bn / add_residual → shape-changing retrofit not safe mid-run
HARNESS_WAIVED_RULES: frozenset[str] = frozenset({"R2", "R6"})

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
