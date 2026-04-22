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
    "OpenAIDecisionPolicy",
    "ScriptedDecisionPolicy",
    "build_decision_messages",
    "build_iterative_system_prompt",
    "playbook_text",
]
