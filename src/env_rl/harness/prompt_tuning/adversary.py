"""Prompt Adversary — generates edge-case scenarios designed to find prompt weaknesses.

Given the current champion prompt, the Adversary produces 2-3 candidate
diagnostic states where the prompt might produce an incorrect decision.
Each candidate is vetted by running the champion on it; scenarios the
champion *fails* are the most valuable — they reveal weaknesses the
existing test suite does not cover, and the next Tuner cycle can target
those specifically.

This is essentially mutation testing for prompts.
"""

from __future__ import annotations

import json
from typing import Any

from env_rl.harness.prompt_tuning.scenarios import (
    ExpectedDecision,
    Scenario,
    ScenarioResult,
)

_ADVERSARY_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "top_rule": {
                        "type": "string",
                        "enum": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
                    },
                    "fired_rules": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
                        },
                        "minItems": 1,
                    },
                    "diagnostic_values": {
                        "type": "object",
                        "properties": {
                            "max_layer_grad_norm": {"type": "number"},
                            "min_layer_grad_norm": {"type": "number"},
                            "dead_relu_fraction": {"type": "number"},
                            "train_loss": {"type": "number"},
                            "val_loss": {"type": "number"},
                            "train_acc": {"type": "number"},
                            "update_to_param_ratio": {"type": "number"},
                            "grad_noise_scale": {"type": "number"},
                        },
                        "required": ["max_layer_grad_norm", "min_layer_grad_norm",
                                     "dead_relu_fraction", "train_loss", "val_loss",
                                     "train_acc", "update_to_param_ratio",
                                     "grad_noise_scale"],
                        "additionalProperties": False,
                    },
                    "expected_event_type": {
                        "type": "string",
                        "enum": ["hyperparameter_change", "architecture_change",
                                 "rule_triggered_no_action"],
                    },
                    "expected_remedy_direction": {
                        "type": "string",
                        "enum": ["decrease_lr", "increase_lr", "stop",
                                 "swap_activation", "add_capacity",
                                 "increase_batch_size", "decrease_batch_size",
                                 "add_bn_or_residual", "none"],
                    },
                    "current_lr": {"type": "number"},
                    "current_batch_size": {"type": "integer"},
                },
                "required": ["name", "description", "top_rule", "fired_rules",
                             "diagnostic_values", "expected_event_type",
                             "expected_remedy_direction", "current_lr",
                             "current_batch_size"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}

_ADVERSARY_SYSTEM = """\
You are an adversarial prompt auditor. Your job is to find edge cases where
a given decision prompt might produce the wrong answer.

You will be shown a system prompt that instructs an LLM on how to respond
when training-diagnostic rules fire. Generate 2 or 3 edge-case diagnostic
scenarios where the prompt is most likely to confuse the model.

Good edge cases include:
  - Tied-precedence situations where multiple rules of the same class fire
  - Just-barely-threshold values where EMA smoothing matters
  - Conflicting signals (e.g. R5 fires but grads are also elevated)
  - Recently-actioned rules that should have cleared but still flag

Return valid JSON matching the schema.
"""


class PromptAdversary:
    """Generates edge-case scenarios that target the current prompt's weak spots."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.9,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def generate_candidates(self, system_prompt_to_audit: str) -> list[Scenario]:
        messages = [
            {"role": "system", "content": _ADVERSARY_SYSTEM},
            {"role": "user", "content": (
                "Here is the prompt to audit. Produce edge cases:\n\n"
                "```\n" + system_prompt_to_audit + "\n```"
            )},
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "adversary_scenarios",
                    "strict": True,
                    "schema": _ADVERSARY_SCHEMA,
                },
            },
        )
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return []
        return [self._to_scenario(d) for d in payload.get("scenarios", [])]

    @staticmethod
    def _to_scenario(d: dict) -> Scenario:
        fired_map = {f"R{i}": False for i in range(1, 8)}
        for r in d.get("fired_rules", []):
            fired_map[r] = True
        metrics_now = dict(d["diagnostic_values"])
        metrics_now["epoch"] = 0
        # Fabricate a 3-epoch history with repeated current values so that
        # EMA would have latched on to the same regime.
        history = [{**metrics_now, "epoch": i} for i in range(3)]
        expected = ExpectedDecision(
            event_types=frozenset({d["expected_event_type"]}),
            cited_rule=d["top_rule"],
            acceptable_remedy_directions=frozenset({d["expected_remedy_direction"]}),
            required_edit_op=None,
        )
        return Scenario(
            name=d["name"],
            description=d["description"],
            metrics_history=history,
            all_fired=fired_map,
            top_rule=d["top_rule"],
            current_lr=float(d["current_lr"]),
            current_batch_size=int(d["current_batch_size"]),
            expected=expected,
        )

    @staticmethod
    def keep_failures(
        champion_results: list[ScenarioResult],
    ) -> list[Scenario]:
        """From a list of scenarios the champion just failed, return the
        Scenario objects to add to the suite."""
        return [r.scenario for r in champion_results if not r.passed]
