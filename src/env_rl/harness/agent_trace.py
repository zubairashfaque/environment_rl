"""Unified agent trace — every agent interaction recorded in one place.

The six agents in env_rl (decision LLM, Tuner, Tester, Judge, Distiller,
Adversary) each do their own job. This module gives them a shared, opt-in
tracing channel so you can replay the *entire session* as a single event
stream without hunting across five different log files.

Usage (dependency injection):

    tracer = AgentTracer(path / "agent_trace.jsonl")
    policy = OpenAIDecisionPolicy(..., tracer=tracer)
    meta_loop = MetaLoop(..., tracer=tracer)

When ``tracer=None`` (default), all record() calls are no-ops — zero cost
for anyone who doesn't want the trace.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class AgentEvent:
    """One observable thing an agent did."""

    ts: float                       # wall-clock unix timestamp
    attempt_index: int | None       # meta-loop attempt, or None if not applicable
    epoch: int | None               # training epoch, or None
    agent: str                      # decision_llm | tuner | tester | judge |
                                    # distiller | adversary | meta_loop
    action: str                     # what this agent just did
    duration_ms: float              # measured latency of the action
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    token_cost: dict[str, int] = field(default_factory=dict)
    model: str | None = None


class AgentTracer:
    """Opt-in JSONL writer for AgentEvent records."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._enabled = enabled and self._path is not None
        if self._enabled:
            assert self._path is not None
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Start fresh each session
            self._path.write_text("")
        self._attempt_index: int | None = None
        self._event_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_attempt_index(self, i: int) -> None:
        """Meta-loop or iterative driver calls this at the start of each attempt."""
        self._attempt_index = i

    def record(
        self,
        *,
        agent: str,
        action: str,
        duration_ms: float = 0.0,
        epoch: int | None = None,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        token_cost: dict[str, int] | None = None,
        model: str | None = None,
    ) -> None:
        if not self._enabled or self._path is None:
            return
        event = AgentEvent(
            ts=time.time(),
            attempt_index=self._attempt_index,
            epoch=epoch,
            agent=agent,
            action=action,
            duration_ms=duration_ms,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            token_cost=token_cost or {},
            model=model,
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        self._event_count += 1

    @contextmanager
    def timed(
        self,
        *,
        agent: str,
        action: str,
        epoch: int | None = None,
        input_summary: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Context manager that auto-times the block and records an event.

        The block can mutate the yielded dict to add output_summary/token_cost
        that'll land in the recorded event:

            with tracer.timed(agent="tuner", action="propose_edit") as out:
                edit = self._tuner.propose_edit(...)
                out["output_summary"] = {"technique": edit.technique}
        """
        out: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield out
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.record(
                agent=agent,
                action=action,
                duration_ms=dt,
                epoch=epoch,
                input_summary=input_summary,
                output_summary=out.get("output_summary"),
                token_cost=out.get("token_cost"),
                model=model,
            )

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def path(self) -> Path | None:
        return self._path


# Convenience: a shared, disabled-by-default null tracer so callers can
# always just write `self._tracer.record(...)` without None-checks.
NULL_TRACER = AgentTracer(path=None, enabled=False)
