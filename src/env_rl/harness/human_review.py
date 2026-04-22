"""Human-in-the-loop review channel.

Let a human look at an attempt's decisions and mark them gold or bad. Gold
decisions become scenario-suite fixtures; bad decisions inform the Tuner's
negative-constraint prompts.

File format: ``decision_review.jsonl`` in the attempt directory, one JSON
object per line, human-owned (git-ignored by default).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from env_rl.harness.prompt_tuning.scenarios import (
    ExpectedDecision,
    Scenario,
)


@dataclass
class HumanReview:
    attempt_dir: str
    epoch: int
    decision: dict[str, Any]   # the logged decision payload
    verdict: str                # "gold" | "bad" | "skip"
    notes: str = ""
    reviewer: str = "user"

    def to_jsonl(self) -> str:
        return json.dumps({
            "attempt_dir": self.attempt_dir,
            "epoch": self.epoch,
            "decision": self.decision,
            "verdict": self.verdict,
            "notes": self.notes,
            "reviewer": self.reviewer,
        })


def append_review(path: Path | str, review: HumanReview) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(review.to_jsonl() + "\n")


def load_reviews(path: Path | str) -> list[HumanReview]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[HumanReview] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(HumanReview(
            attempt_dir=d.get("attempt_dir", ""),
            epoch=int(d.get("epoch", 0)),
            decision=d.get("decision", {}),
            verdict=str(d.get("verdict", "skip")),
            notes=str(d.get("notes", "")),
            reviewer=str(d.get("reviewer", "user")),
        ))
    return out


def load_all_reviews(base_dir: Path | str) -> list[HumanReview]:
    """Walk every attempt_* folder in base_dir and concatenate reviews."""
    base = Path(base_dir)
    if not base.exists():
        return []
    all_reviews: list[HumanReview] = []
    for attempt in sorted(base.glob("attempt_*")):
        reviews_path = attempt / "decision_review.jsonl"
        all_reviews.extend(load_reviews(reviews_path))
    return all_reviews


def reviews_to_scenarios(reviews: list[HumanReview]) -> list[Scenario]:
    """Convert gold-verdict reviews into Scenario objects.

    Only ``verdict == "gold"`` decisions become scenario fixtures. Bad and
    skip decisions are ignored by the converter but remain in the review
    log for the Tuner to consult.
    """
    out: list[Scenario] = []
    for r in reviews:
        if r.verdict != "gold":
            continue
        decision = r.decision
        cites = decision.get("cites", [])
        if not cites:
            continue
        rule = cites[0]
        evals = {f"R{i}": False for i in range(1, 8)}
        evals[rule] = True
        # We do not have full metrics from the review alone — use a minimal
        # stub so the scenario still exercises the LLM's decision shape.
        metrics = {"epoch": r.epoch}
        history = [metrics] * 3
        expected = ExpectedDecision(
            event_types=frozenset({decision.get("event_type", "rule_triggered_no_action")}),
            cited_rule=rule,
            acceptable_remedy_directions=frozenset({
                decision.get("remedy_direction") or "none"
            }),
            required_edit_op=(decision.get("edit", {}) or {}).get("op"),
        )
        out.append(Scenario(
            name=f"gold_{r.attempt_dir}_{r.epoch}_{rule}",
            description=f"Human-marked gold decision from {r.attempt_dir} ep{r.epoch}",
            metrics_history=history,
            all_fired=evals,
            top_rule=rule,
            current_lr=0.0,
            current_batch_size=0,
            expected=expected,
        ))
    return out
