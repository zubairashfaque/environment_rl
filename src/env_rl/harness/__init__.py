"""In-context iterative improvement harness.

Plugs a real LLM (OpenAI-compatible) into the environment as the
decision-maker for rule firings. "Learning" happens by replaying prior
attempts' scores and violations into the next attempt's system prompt —
model weights never change. Documented precisely as *iterative self-refine*,
not RL.
"""

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
    "OpenAIDecisionPolicy",
    "ScriptedDecisionPolicy",
    "build_decision_messages",
    "build_iterative_system_prompt",
    "playbook_text",
]
