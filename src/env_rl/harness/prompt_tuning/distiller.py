"""Prompt Distiller — compress a bloated prompt while preserving scenario pass rate.

After many meta-loop iterations, the champion prompt accumulates negative
constraints, one-shot examples, and chain-of-thought instructions. The
Distiller asks the LLM to produce a terser version that keeps every
semantic constraint intact. The Tester validates; the Judge decides whether
to promote.

A distillation only gets promoted when:
  - pass rate on the suite is ≥ old pass rate (no regressions allowed)
  - new prompt is at least 15% shorter

Otherwise the bloated prompt is kept — shrinking at the cost of correctness
is never worth it.
"""

from __future__ import annotations

from typing import Any

from env_rl.harness.prompt_tuning.tuner import PromptEdit


_DISTILL_SYSTEM = """\
You are a prompt-engineering expert whose only job is to compress system
prompts without changing their meaning. Read the prompt below carefully.

Produce a new prompt that:
  - preserves EVERY rule, instruction, constraint, example, and piece of
    metadata present in the original
  - merges redundant sentences
  - uses shorter phrasing where it does not lose clarity
  - keeps all JSON-shape descriptions, enum values, field names EXACTLY
  - keeps all numbered lists numbered
  - preserves all worked examples verbatim (these are functional)
  - does not summarize or paraphrase enum values

Return ONLY the compressed prompt text. No preamble, no trailing commentary.
"""


class PromptDistiller:
    """Compresses a prompt while preserving scenario pass rate."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        min_reduction_ratio: float = 0.15,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._min_reduction_ratio = min_reduction_ratio

    def distill(self, prompt: str) -> PromptEdit:
        """Ask the LLM for a compressed version.

        Returns a ``PromptEdit`` whose ``addition`` is the COMPLETE new
        prompt (not an additive fragment) — this is a special case that
        replaces the whole prompt when applied.
        """
        messages = [
            {"role": "system", "content": _DISTILL_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        compressed = response.choices[0].message.content or ""
        reduction = 1 - len(compressed) / max(1, len(prompt))
        return PromptEdit(
            technique="distillation",
            rationale=(
                f"distilled prompt: {len(prompt)} -> {len(compressed)} chars "
                f"({reduction:.1%} shorter)"
            ),
            addition=compressed,  # full replacement, see DistillationEdit.apply below
        )


class DistillationEdit(PromptEdit):
    """Distillation is a full replacement, not an appended addition.

    The standard ``PromptEdit.apply`` appends; distilled prompts replace.
    Callers should check ``edit.technique == "distillation"`` and use the
    addition directly when replacing.
    """

    def apply(self, base_prompt: str) -> str:  # noqa: ARG002 — intentional
        return self.addition
