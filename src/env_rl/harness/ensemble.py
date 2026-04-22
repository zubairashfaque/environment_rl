"""Ensemble voting for high-stakes decisions (stability-class rules).

When a stability rule (R7 or R6) fires, the cost of picking the wrong remedy
is high — a bad call here can crash the run. Rather than trust a single
sample, sample the decision N times and take majority vote over event_type
and remedy_direction.

This burns more tokens but improves calibration on the decisions that
matter most. Activated per-call at the harness level via a config flag.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any


def should_ensemble(top_rule: str, stability_rules: frozenset[str] = frozenset({"R6", "R7"})) -> bool:
    return top_rule in stability_rules


@dataclass
class EnsembleResult:
    majority_decision: dict[str, Any]
    agreement: float  # fraction of samples that agreed with majority
    samples: list[dict[str, Any]]


def ensemble_decide(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    json_schema_response_format: dict[str, Any],
    n_samples: int = 3,
    temperature: float = 0.7,
) -> EnsembleResult:
    """Run N samples at higher temperature; return majority-voted decision."""
    samples: list[dict[str, Any]] = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=json_schema_response_format,
        )
        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
            samples.append(parsed)
        except (json.JSONDecodeError, TypeError):
            continue

    if not samples:
        # Return empty decision shell; caller can treat as failure.
        return EnsembleResult(
            majority_decision={}, agreement=0.0, samples=[]
        )

    # Vote on the (event_type, first_cite, remedy_direction) tuple.
    def _key(s: dict[str, Any]) -> tuple:
        return (
            s.get("event_type", ""),
            (s.get("cites", ["?"])[0] if s.get("cites") else "?"),
            s.get("remedy_direction", ""),
        )

    counter = Counter(_key(s) for s in samples)
    (winning_key, winning_count) = counter.most_common(1)[0]
    # Return the first sample that matches the winning key (preserves the
    # fullest JSON shape — remedy_params etc).
    for s in samples:
        if _key(s) == winning_key:
            return EnsembleResult(
                majority_decision=s,
                agreement=winning_count / len(samples),
                samples=samples,
            )
    # Unreachable but keep mypy happy
    return EnsembleResult(
        majority_decision=samples[0], agreement=1.0 / len(samples), samples=samples
    )
