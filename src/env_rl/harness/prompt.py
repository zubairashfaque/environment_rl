"""Prompt construction for the LLM decision agent.

Four things to assemble per call:
  1. System prompt — playbook text + decision schema + response rules
  2. Iterative feedback — prior attempts' scores and violations (optional)
  3. Per-decision user prompt — current epoch, metrics, fired rules
  4. JSON schema for OpenAI structured-output validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PLAYBOOK_PATH = Path(__file__).resolve().parents[3] / "docs" / "playbook.md"


def playbook_text() -> str:
    """Return the contents of ``docs/playbook.md`` as a string."""
    return _PLAYBOOK_PATH.read_text()


# -- JSON schema -------------------------------------------------------------

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": [
                "hyperparameter_change",
                "architecture_change",
                "rule_triggered_no_action",
            ],
        },
        "cites": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
            },
            "minItems": 1,
            "maxItems": 1,
        },
        "justification": {"type": "string"},
        "remedy_direction": {
            "type": "string",
            "enum": [
                "decrease_lr",
                "increase_lr",
                "decrease_batch_size",
                "increase_batch_size",
                "add_capacity",
                "swap_activation",
                "add_bn_or_residual",
                "stop",
                "none",
            ],
        },
        "remedy_params": {
            "type": "object",
            "properties": {
                "lr_new": {"type": "number"},
                "edit_op": {
                    "type": "string",
                    "enum": ["swap_activation", "add_block", "remove_block", "none"],
                },
                "edit_to": {
                    "type": "string",
                    "enum": ["relu", "leaky_relu", "gelu", "none"],
                },
            },
            "required": ["lr_new", "edit_op", "edit_to"],
            "additionalProperties": False,
        },
    },
    "required": [
        "event_type",
        "cites",
        "justification",
        "remedy_direction",
        "remedy_params",
    ],
    "additionalProperties": False,
}


# -- System prompt -----------------------------------------------------------

_RESPONSE_RULES = """\
You are the decision-maker for a PyTorch CNN training run on CIFAR-10, scored
by an automated judge on TWO INDEPENDENT AXES: final-model accuracy AND
process discipline. You cannot trade one for the other.

When a rule fires from the 7-rule playbook, you respond with ONE decision, in
this JSON shape:

{
  "event_type": "hyperparameter_change" | "architecture_change" | "rule_triggered_no_action",
  "cites": ["R<N>"],
  "justification": "short text explaining why this action fits the playbook",
  "remedy_direction": "decrease_lr" | "increase_lr" | ...,
  "remedy_params": { "lr_new": <float>, "edit_op": <op>, "edit_to": <str> }
}

Hard rules (violations lose process score, but they are caught):
  - `cites` must list the rule that actually fired (exactly one).
  - Conflict precedence when multiple rules fire: stability > capacity > tuning > process.
    (R7/R6 beat R4/R5 beat R1/R2 beat R3.) Action the highest-precedence rule;
    defer the others explicitly in subsequent calls.
  - event_type must match the rule's class:
      R1/R2/R3 → hyperparameter_change (or rule_triggered_no_action)
      R4/R5    → architecture_change (or rule_triggered_no_action)
      R6/R7    → hyperparameter_change OR architecture_change (or no-action)
  - remedy_direction must match the playbook remedy sign:
      R7 → decrease_lr
      R1 (high ratio) → decrease_lr
      R1 (low ratio)  → increase_lr
      R2 (low GNS)    → increase_batch_size
      R2 (high GNS)   → decrease_batch_size
      R3 → stop
      R5 → swap_activation
      R4 → add_capacity
      R6 → add_bn_or_residual
  - Fill remedy_params even if unused. For actions that do not change LR, set
    lr_new to the CURRENT lr. For actions that are not architecture changes,
    set edit_op to "none" and edit_to to "none".

  HARNESS CAPABILITIES (every rule below is fully actionable; every
  violation costs process score):
    R1 (LR change)        → hyperparameter_change with lr_new
    R2 (batch size)       → hyperparameter_change with
                             remedy_direction="increase_batch_size" or
                             "decrease_batch_size" (harness rebuilds loader)
    R3 (early stop)       → hyperparameter_change with
                             remedy_direction="stop" (loop exits cleanly)
    R4 (add capacity)     → architecture_change with edit_op="add_block"
    R5 (activations)      → architecture_change with
                             edit_op="swap_activation", edit_to in
                             {"leaky_relu", "gelu"}
    R6 (vanishing grads)  → hyperparameter_change with lr_new (LR warmup
                             helps vanishing gradients indirectly per playbook)
    R7 (exploding)        → hyperparameter_change with lr_new (LR drop 10x)

  Every rule firing must produce a matching decision within the ±2-epoch
  window; deferrals must eventually clear; the wrong remedy direction is
  a process violation. No rule is waived.
  - Justification should be concise (under 40 words) and cite the relevant
    diagnostic signal (e.g., "max_layer_grad_norm EMA 14.2 over 3 epochs").
  - `rule_triggered_no_action` is a deliberate deferral. Only use it when you
    have a specific reason to wait (e.g., "deferred_to_R7" when R7 takes
    precedence). The deferred rule MUST clear later — if it doesn't, it's a
    violation.

Do NOT output anything other than the JSON object.
"""


def build_iterative_system_prompt(
    prior_attempts: list[AttemptSummary] | None = None,
    *,
    include_playbook: bool = True,
) -> str:
    """System prompt with the playbook text and (optionally) prior-attempt feedback."""
    parts: list[str] = [_RESPONSE_RULES]
    if include_playbook:
        parts.append("\n\n=== THE PLAYBOOK ===\n\n" + playbook_text())
    if prior_attempts:
        parts.append("\n\n=== PRIOR ATTEMPTS (LEARN FROM THESE) ===\n")
        for i, a in enumerate(prior_attempts, start=1):
            parts.append(a.to_prompt_block(index=i))
        parts.append(
            "\nAnalyze the pattern of violations and avoid repeating them. If a "
            "prior decision was flagged for event_type mismatch, remedy_direction "
            "mismatch, or precedence error, make sure your decisions in this "
            "attempt do NOT repeat that pattern."
        )
    return "".join(parts)


# -- Per-decision user prompt ------------------------------------------------


def build_decision_messages(
    *,
    system_prompt: str,
    epoch: int,
    top_rule: str,
    all_fired: dict[str, bool],
    metrics: dict[str, Any],
    current_lr: float,
    current_batch_size: int,
    recent_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the ``messages=[...]`` argument for the OpenAI API."""
    fired = sorted(r for r, v in all_fired.items() if v)
    key_metrics = {
        k: metrics[k]
        for k in (
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "train_acc",
            "max_layer_grad_norm",
            "min_layer_grad_norm",
            "dead_relu_fraction",
            "update_to_param_ratio",
            "grad_noise_scale",
        )
        if k in metrics
    }
    recent = (recent_history or [])[-3:]
    user = (
        f"Epoch {epoch}. Rule(s) fired: {fired}. "
        f"You must action the highest-precedence rule, which is {top_rule}.\n\n"
        f"Current hyperparameters:\n"
        f"  lr = {current_lr}\n"
        f"  batch_size = {current_batch_size}\n\n"
        f"Current-epoch diagnostics:\n"
    )
    for k, v in key_metrics.items():
        user += f"  {k} = {v}\n"
    if recent:
        user += "\nRecent epochs (for EMA/trend context):\n"
        for h in recent:
            user += (
                f"  epoch {h.get('epoch')}: "
                f"max_grad={h.get('max_layer_grad_norm', 0):.4g}, "
                f"min_grad={h.get('min_layer_grad_norm', 0):.4g}, "
                f"dead_relu={h.get('dead_relu_fraction', 0):.4g}, "
                f"val_loss={h.get('val_loss', 0):.4g}\n"
            )
    user += (
        f"\nRespond with ONE JSON decision citing {top_rule}. "
        f"All other fired rules should be deferred via a separate call "
        f"(the driver handles that). Do not repeat prior violations."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


# -- Prior-attempt summary ---------------------------------------------------


@dataclass
class AttemptSummary:
    """One prior attempt's scores and violations, for iterative feedback."""

    attempt_index: int
    accuracy_score: float
    process_score: float
    test_accuracy: float
    total_decisions: int
    violations: int
    violation_summary: list[dict[str, Any]]

    def to_prompt_block(self, *, index: int) -> str:
        lines = [
            f"\nAttempt {index}:",
            f"  accuracy_score = {self.accuracy_score:.3f} "
            f"(test_accuracy = {self.test_accuracy:.3f})",
            f"  process_score  = {self.process_score:.3f} "
            f"({self.violations} violations over {self.total_decisions} decisions)",
        ]
        if self.violation_summary:
            lines.append("  specific violations:")
            for v in self.violation_summary[:8]:
                lines.append(
                    f"    - {v.get('kind')} (rule={v.get('rule')}, "
                    f"epoch={v.get('epoch')}): {v.get('detail', '')[:140]}"
                )
        return "\n".join(lines) + "\n"
