"""Pluggable decision policies.

``DecisionPolicy`` is a small protocol: given the current epoch, fired rules,
and metrics, return one ``Decision``. Two implementations ship:

  - ``ScriptedDecisionPolicy`` — the hard-coded heuristics from the original
    reference run (no LLM).
  - ``OpenAIDecisionPolicy`` — calls the OpenAI API with structured output
    and a system prompt that carries prior-attempt feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from env_rl.harness.prompt import (
    DECISION_SCHEMA,
    AttemptSummary,
    build_decision_messages,
    build_iterative_system_prompt,
)


@dataclass
class Decision:
    """The single structured action logged via ``monitor.log_decision``."""

    event_type: str
    cites: list[str]
    justification: str
    remedy_direction: str | None = None
    remedy_params: dict[str, Any] = field(default_factory=dict)

    def to_log_kwargs(self) -> dict[str, Any]:
        """Shape ready for ``monitor.log_decision(**kwargs)``."""
        out: dict[str, Any] = {
            "event_type": self.event_type,
            "cites": list(self.cites),
            "justification": self.justification,
        }
        if self.remedy_direction:
            out["remedy_direction"] = self.remedy_direction
        if self.remedy_params:
            out.update(self.remedy_params)
        return out


class DecisionPolicy(Protocol):
    """Contract every policy must satisfy."""

    def decide(
        self,
        *,
        top_rule: str,
        all_fired: dict[str, bool],
        metrics: dict[str, Any],
        epoch: int,
        current_lr: float,
        current_batch_size: int,
        recent_history: list[dict[str, Any]],
    ) -> Decision: ...


# ---------------------------------------------------------------------------
# Scripted (the original hard-coded heuristic)
# ---------------------------------------------------------------------------


class ScriptedDecisionPolicy:
    """Non-learning baseline. Mirrors the original reference_run heuristic."""

    def decide(
        self,
        *,
        top_rule: str,
        all_fired: dict[str, bool],
        metrics: dict[str, Any],
        epoch: int,
        current_lr: float,
        current_batch_size: int,
        recent_history: list[dict[str, Any]],
    ) -> Decision:
        if top_rule == "R7":
            return Decision(
                "hyperparameter_change",
                ["R7"],
                f"grad norm EMA over threshold at epoch {epoch}; drop LR /10 per R7",
                remedy_direction="decrease_lr",
                remedy_params={"lr_new": current_lr / 10},
            )
        if top_rule == "R6":
            return Decision(
                "hyperparameter_change",
                ["R6"],
                f"vanishing grads at epoch {epoch}; warmup schedule per R6",
                remedy_direction="add_bn_or_residual",
            )
        if top_rule == "R1":
            return Decision(
                "hyperparameter_change",
                ["R1"],
                f"update-to-param ratio out of band at epoch {epoch}; reduce LR",
                remedy_direction="decrease_lr",
                remedy_params={"lr_new": current_lr / 3},
            )
        if top_rule == "R2":
            return Decision(
                "hyperparameter_change",
                ["R2"],
                f"grad noise scale out of band at epoch {epoch}; adjust batch",
                remedy_direction="increase_batch_size",
            )
        if top_rule == "R3":
            return Decision(
                "hyperparameter_change",
                ["R3"],
                f"val loss plateau at epoch {epoch}; early stop",
                remedy_direction="stop",
            )
        if top_rule == "R5":
            return Decision(
                "architecture_change",
                ["R5"],
                f"dead-ReLU EMA above threshold at epoch {epoch}; swap activation",
                remedy_direction="swap_activation",
                remedy_params={
                    "edit": {"op": "swap_activation", "to": "leaky_relu"},
                },
            )
        if top_rule == "R4":
            return Decision(
                "architecture_change",
                ["R4"],
                f"train-acc plateau without grad issues at epoch {epoch}; add capacity",
                remedy_direction="add_capacity",
                remedy_params={"edit": {"op": "add_block"}},
            )
        # fallback
        return Decision(
            "rule_triggered_no_action",
            [top_rule],
            f"deferred {top_rule}",
        )


# ---------------------------------------------------------------------------
# OpenAI (in-context iterative)
# ---------------------------------------------------------------------------


class OpenAIDecisionPolicy:
    """Calls the OpenAI Chat Completions API with JSON-schema structured output."""

    def __init__(
        self,
        *,
        client: Any,  # openai.OpenAI or any object with .chat.completions.create
        model: str = "gpt-4o-mini",
        prior_attempts: list[AttemptSummary] | None = None,
        temperature: float = 0.2,
        system_prompt: str | None = None,
        transcript_path: Any = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt or build_iterative_system_prompt(
            prior_attempts or []
        )
        self._transcript_path = transcript_path
        # Write the system prompt once at the top of the transcript so every
        # subsequent call record can stay small.
        if transcript_path is not None:
            import json as _json
            from pathlib import Path as _Path
            p = _Path(transcript_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(_json.dumps({
                    "kind": "system_prompt",
                    "model": model,
                    "temperature": temperature,
                    "text": self._system_prompt,
                }) + "\n")

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def decide(
        self,
        *,
        top_rule: str,
        all_fired: dict[str, bool],
        metrics: dict[str, Any],
        epoch: int,
        current_lr: float,
        current_batch_size: int,
        recent_history: list[dict[str, Any]],
    ) -> Decision:
        messages = build_decision_messages(
            system_prompt=self._system_prompt,
            epoch=epoch,
            top_rule=top_rule,
            all_fired=all_fired,
            metrics=metrics,
            current_lr=current_lr,
            current_batch_size=current_batch_size,
            recent_history=recent_history,
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
        if self._transcript_path is not None:
            from pathlib import Path as _Path
            p = _Path(self._transcript_path)
            # Extract usage if the SDK provided it (openai >= 1.0)
            usage: dict[str, Any] = {}
            u = getattr(response, "usage", None)
            if u is not None:
                usage = {
                    "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
                }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "call",
                    "epoch": epoch,
                    "top_rule": top_rule,
                    "all_fired": {k: bool(v) for k, v in all_fired.items()},
                    "user_message": messages[1]["content"],
                    "response": raw,
                    "usage": usage,
                    "model": self._model,
                }) + "\n")
        data = json.loads(raw)
        return _decision_from_dict(data, top_rule=top_rule, current_lr=current_lr)


def _decision_from_dict(
    d: dict[str, Any], *, top_rule: str, current_lr: float
) -> Decision:
    event_type = str(d["event_type"])
    cites = list(d.get("cites", [top_rule])) or [top_rule]
    justification = str(d.get("justification", ""))
    remedy_direction = d.get("remedy_direction")
    remedy_params_raw = d.get("remedy_params", {}) or {}

    # Translate the schema-locked remedy_params into what monitor.log_decision + the
    # training-loop remedy-applier expect.
    remedy_params: dict[str, Any] = {}
    lr_new = remedy_params_raw.get("lr_new")
    if lr_new is not None and lr_new != current_lr and lr_new > 0:
        remedy_params["lr_new"] = float(lr_new)
    edit_op = remedy_params_raw.get("edit_op")
    if edit_op and edit_op != "none":
        edit: dict[str, Any] = {"op": edit_op}
        edit_to = remedy_params_raw.get("edit_to")
        if edit_to:
            edit["to"] = edit_to
        remedy_params["edit"] = edit
    return Decision(
        event_type=event_type,
        cites=cites,
        justification=justification,
        remedy_direction=remedy_direction if remedy_direction != "none" else None,
        remedy_params=remedy_params,
    )
