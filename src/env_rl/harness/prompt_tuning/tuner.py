"""Prompt Tuner — proposes a prompt edit based on observed violations.

The Tuner does not call the LLM itself to rewrite the prompt; that would be
high-variance and unstable. Instead it has a small library of well-defined
*techniques* (negative-constraint, one-shot, few-shot, CoT, role-play) and
picks the smallest edit that plausibly addresses the observed violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


PROMPT_TECHNIQUES: tuple[str, ...] = (
    "baseline",
    "negative_constraint",
    "one_shot",
    "few_shot",
    "chain_of_thought",
    "role_play",
    "structured_template",
)


@dataclass
class PromptEdit:
    """Describes a proposed change to the prompt."""

    technique: str
    rationale: str
    addition: str  # text appended to the base prompt
    target_violation_kinds: frozenset[str] = field(default_factory=frozenset)

    def apply(self, base_prompt: str) -> str:
        if not self.addition.strip():
            return base_prompt
        return base_prompt + "\n\n" + self.addition.strip() + "\n"


# ---------------------------------------------------------------------------
# Technique templates
# ---------------------------------------------------------------------------


_ROLE_PLAY_ADDITION = """\
You are a senior ML engineer with fifteen years of experience training CNNs.
You have seen thousands of runs fail at exactly the patterns the playbook
describes. Read the diagnostics carefully before answering. The cost of a
wrong remedy is higher than the cost of a careful deferral.
"""

_CHAIN_OF_THOUGHT_ADDITION = """\
Before returning the JSON decision, reason in this order:
  1. List every fired rule and its precedence class.
  2. Pick the highest-precedence class; if tied, pick the lowest-numbered rule.
  3. Read the playbook's remedy for that rule.
  4. Choose event_type and remedy_direction to match the remedy.
  5. Verify the justification cites a specific diagnostic value.
Only after these five checks, emit the JSON.
"""

_STRUCTURED_TEMPLATE_ADDITION = """\
Use this 4-part template in your justification:
  Symptom: <the specific diagnostic value out of band>
  Rule: <Rx> per the playbook
  Remedy: <what the playbook prescribes>
  Evidence: <the EMA value or trend across the last 3 epochs>
"""


def _one_shot_example(rule: str) -> str:
    # A single fully-worked example for the rule most frequently mis-actioned.
    examples = {
        "R7": """\
Example:
  Epoch 3. R7 fires. max_layer_grad_norm EMA 14.2 over 3 epochs.
  Response:
  {
    "event_type": "hyperparameter_change",
    "cites": ["R7"],
    "justification": "max_layer_grad_norm EMA 14.2 over 3 consecutive epochs exceeds R7 threshold 10; drop LR 10x per playbook",
    "remedy_direction": "decrease_lr",
    "remedy_params": {"lr_new": 0.005, "edit_op": "none", "edit_to": "none"}
  }
""",
        "R5": """\
Example:
  Epoch 4. R5 fires. dead_relu_fraction EMA 0.71 over 3 epochs.
  Response:
  {
    "event_type": "architecture_change",
    "cites": ["R5"],
    "justification": "dead_relu_fraction EMA 0.71 exceeds R5 threshold 0.40 for 3 epochs; swap to leaky_relu",
    "remedy_direction": "swap_activation",
    "remedy_params": {"lr_new": 0.05, "edit_op": "swap_activation", "edit_to": "leaky_relu"}
  }
""",
        "R4": """\
Example:
  Epoch 10. R4 fires. train_acc 0.70 for 3 epochs, grads clean, dead_relu near zero.
  Response:
  {
    "event_type": "architecture_change",
    "cites": ["R4"],
    "justification": "train_acc plateau at 0.70 for 3 epochs with clean gradients (max 0.3) and 0% dead ReLU; out of capacity — add block per R4",
    "remedy_direction": "add_capacity",
    "remedy_params": {"lr_new": 0.05, "edit_op": "add_block", "edit_to": "none"}
  }
""",
        "R3": """\
Example:
  Epoch 22. R3 fires. val_loss flat 1.15 for 5 epochs.
  Response:
  {
    "event_type": "hyperparameter_change",
    "cites": ["R3"],
    "justification": "val_loss plateau at 1.15 across 5 epochs exceeds R3 patience; stop and save best checkpoint",
    "remedy_direction": "stop",
    "remedy_params": {"lr_new": 0.05, "edit_op": "none", "edit_to": "none"}
  }
""",
    }
    return examples.get(rule, "")


# ---------------------------------------------------------------------------
# The Tuner itself
# ---------------------------------------------------------------------------


class PromptTuner:
    """Picks the smallest useful edit to apply to the current prompt."""

    def propose_edit(
        self,
        *,
        violations: list[dict],
        attempt_index: int,
    ) -> PromptEdit:
        """Return a single edit. Escalate through techniques with violation count.

        Strategy:
          - 0 violations -> baseline (no edit)
          - 1-2 violations -> negative_constraint (cheapest)
          - 3-5 violations -> one_shot targeting the most-violated rule
          - 6-10 violations -> few_shot covering top 3 rules
          - >10 violations -> chain_of_thought + few_shot
          - on repeat offenses across attempts -> role_play + structured_template
        """
        if not violations:
            return PromptEdit(
                technique="baseline",
                rationale="No violations to address",
                addition="",
            )

        # What rule(s) are most often flagged?
        rule_counts: dict[str, int] = {}
        kind_counts: dict[str, int] = {}
        for v in violations:
            rule_counts[v.get("rule", "?")] = rule_counts.get(v.get("rule", "?"), 0) + 1
            kind_counts[v.get("kind", "?")] = kind_counts.get(v.get("kind", "?"), 0) + 1
        top_rule = max(rule_counts, key=rule_counts.get)
        total = len(violations)

        if total <= 2:
            return self._negative_constraint(top_rule, kind_counts)
        if total <= 5:
            return self._one_shot(top_rule, violations)
        if total <= 10:
            return self._few_shot(rule_counts)
        if attempt_index >= 3:
            return self._role_play_plus_template()
        return self._chain_of_thought()

    # ---- technique builders ----

    def _negative_constraint(
        self, top_rule: str, kinds: dict[str, int]
    ) -> PromptEdit:
        tips = []
        if kinds.get("unresolved_deferral", 0):
            tips.append(
                f"- Do NOT defer {top_rule} indefinitely. "
                f"If {top_rule} keeps firing across 3+ epochs, action it directly."
            )
        if kinds.get("precedence_violation", 0):
            tips.append(
                "- When two rules of the same precedence class fire, action the "
                "lower-numbered rule first (R4 before R5, R1 before R2)."
            )
        if kinds.get("event_type_mismatch", 0):
            tips.append(
                f"- Double-check event_type: {top_rule} should map to the class "
                "specified in the playbook; no other event_type is acceptable."
            )
        if kinds.get("direction_mismatch", 0):
            tips.append(
                f"- remedy_direction for {top_rule} must match the playbook; "
                "consult the playbook text before emitting the decision."
            )
        if not tips:
            tips.append(
                f"- Pay special attention to {top_rule}; recent attempts "
                "violated its handling contract."
            )
        addition = "Lessons from prior attempts — avoid these:\n" + "\n".join(tips)
        return PromptEdit(
            technique="negative_constraint",
            rationale=f"{len(tips)} constraint(s) targeting {top_rule}",
            addition=addition,
            target_violation_kinds=frozenset(kinds),
        )

    def _one_shot(self, top_rule: str, violations: list[dict]) -> PromptEdit:
        example = _one_shot_example(top_rule)
        if not example:
            # Fall back to negative constraint if no example exists
            return self._negative_constraint(top_rule, {"?": len(violations)})
        return PromptEdit(
            technique="one_shot",
            rationale=f"One-shot example for most-violated rule {top_rule}",
            addition=example,
        )

    def _few_shot(self, rule_counts: dict[str, int]) -> PromptEdit:
        top3 = sorted(rule_counts.items(), key=lambda kv: -kv[1])[:3]
        parts = [
            _one_shot_example(rule) for rule, _ in top3 if _one_shot_example(rule)
        ]
        if not parts:
            return PromptEdit(technique="baseline", rationale="No examples", addition="")
        return PromptEdit(
            technique="few_shot",
            rationale=f"Few-shot examples for top {len(parts)} violated rules",
            addition="Here are worked examples of correct decisions:\n\n" + "\n".join(parts),
        )

    def _chain_of_thought(self) -> PromptEdit:
        return PromptEdit(
            technique="chain_of_thought",
            rationale="High violation count — enforce step-by-step reasoning",
            addition=_CHAIN_OF_THOUGHT_ADDITION,
        )

    def _role_play_plus_template(self) -> PromptEdit:
        return PromptEdit(
            technique="role_play",
            rationale="Persistent violations across attempts — raise stakes via persona",
            addition=_ROLE_PLAY_ADDITION + "\n" + _STRUCTURED_TEMPLATE_ADDITION,
        )
