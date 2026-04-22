"""Prompt Tester — evaluates a candidate system prompt on the scenario suite.

Runs one LLM call per scenario with the candidate prompt, then checks the
response against the scenario's expected decision shape.
"""

from __future__ import annotations

import json
from typing import Any

from env_rl.harness.prompt import (
    DECISION_SCHEMA,
    build_decision_messages,
)
from env_rl.harness.prompt_tuning.scenarios import (
    SCENARIO_SUITE,
    Scenario,
    ScenarioResult,
)


class PromptTester:
    """Runs a prompt against the scenario suite and scores it."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def run_suite(
        self,
        system_prompt: str,
        *,
        scenarios: list[Scenario] | None = None,
    ) -> list[ScenarioResult]:
        scenarios = scenarios or SCENARIO_SUITE
        results: list[ScenarioResult] = []
        for scn in scenarios:
            result = self._run_one(system_prompt, scn)
            results.append(result)
        return results

    def _run_one(self, system_prompt: str, scn: Scenario) -> ScenarioResult:
        metrics_now = scn.metrics_history[-1]
        messages = build_decision_messages(
            system_prompt=system_prompt,
            epoch=int(metrics_now.get("epoch", 0)),
            top_rule=scn.top_rule,
            all_fired=scn.all_fired,
            metrics=metrics_now,
            current_lr=scn.current_lr,
            current_batch_size=scn.current_batch_size,
            recent_history=list(scn.metrics_history[-3:]),
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "strict": True,
                    "schema": DECISION_SCHEMA,
                },
            },
        )
        raw = response.choices[0].message.content
        try:
            decision = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return ScenarioResult(
                scenario=scn,
                llm_decision={},
                passed=False,
                failure_reasons=["response was not valid JSON"],
            )
        ok, reasons = self._check(decision, scn)
        return ScenarioResult(
            scenario=scn, llm_decision=decision, passed=ok, failure_reasons=reasons,
        )

    @staticmethod
    def _check(decision: dict, scn: Scenario) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        exp = scn.expected
        if decision.get("event_type") not in exp.event_types:
            reasons.append(
                f"event_type={decision.get('event_type')!r} not in {sorted(exp.event_types)}"
            )
        cites = decision.get("cites", [])
        if exp.cited_rule not in cites:
            reasons.append(f"missing cite for {exp.cited_rule}")
        direction = decision.get("remedy_direction")
        if direction not in exp.acceptable_remedy_directions:
            reasons.append(
                f"remedy_direction={direction!r} not in "
                f"{sorted(exp.acceptable_remedy_directions)}"
            )
        if exp.required_edit_op is not None:
            actual_op = decision.get("remedy_params", {}).get("edit_op")
            if actual_op != exp.required_edit_op:
                reasons.append(
                    f"edit_op={actual_op!r}, expected {exp.required_edit_op!r}"
                )
        return (not reasons, reasons)


def pass_rate(results: list[ScenarioResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)
