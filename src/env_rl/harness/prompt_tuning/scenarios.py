"""Hand-crafted scenarios for stress-testing decision prompts.

Each scenario is a ``(metrics_history, all_fired, top_rule, current_lr,
current_batch_size, expected)`` tuple. ``expected`` lists the *acceptable*
decisions — a prompt passes if the LLM's response falls inside that set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExpectedDecision:
    """Describes an acceptable decision shape — not a single exact answer.

    Any of the allowed ``event_type``/``cites``/``remedy_direction`` combos
    is considered correct. ``required_edit_op`` locks the architecture op
    when the rule mandates one.
    """

    event_types: frozenset[str]
    cited_rule: str
    acceptable_remedy_directions: frozenset[str]
    required_edit_op: str | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    metrics_history: list[dict[str, Any]]
    all_fired: dict[str, bool]
    top_rule: str
    current_lr: float
    current_batch_size: int
    expected: ExpectedDecision


@dataclass
class ScenarioResult:
    scenario: Scenario
    llm_decision: dict[str, Any]
    passed: bool
    failure_reasons: list[str]


# ---------------------------------------------------------------------------
# The suite — one scenario per rule + a handful of adversarial edge cases
# ---------------------------------------------------------------------------


def _fired_only(rule: str) -> dict[str, bool]:
    evals = {f"R{i}": False for i in range(1, 8)}
    evals[rule] = True
    return evals


SCENARIO_SUITE: list[Scenario] = [
    Scenario(
        name="r7_exploding_drop_lr",
        description="R7 fires alone — classic exploding-gradient case",
        metrics_history=[
            {"epoch": 0, "max_layer_grad_norm": 14.0, "min_layer_grad_norm": 0.5,
             "train_loss": 2.1, "val_loss": 2.1},
            {"epoch": 1, "max_layer_grad_norm": 14.2, "min_layer_grad_norm": 0.5,
             "train_loss": 2.0, "val_loss": 2.0},
            {"epoch": 2, "max_layer_grad_norm": 14.5, "min_layer_grad_norm": 0.5,
             "train_loss": 2.0, "val_loss": 2.0},
        ],
        all_fired=_fired_only("R7"),
        top_rule="R7",
        current_lr=0.3,
        current_batch_size=128,
        expected=ExpectedDecision(
            event_types=frozenset({"hyperparameter_change"}),
            cited_rule="R7",
            acceptable_remedy_directions=frozenset({"decrease_lr"}),
        ),
    ),
    Scenario(
        name="r5_dead_relu_swap_activation",
        description="R5 fires — half the neurons are dead",
        metrics_history=[
            {"epoch": 0, "dead_relu_fraction": 0.68, "max_layer_grad_norm": 0.5},
            {"epoch": 1, "dead_relu_fraction": 0.71, "max_layer_grad_norm": 0.5},
            {"epoch": 2, "dead_relu_fraction": 0.73, "max_layer_grad_norm": 0.5},
        ],
        all_fired=_fired_only("R5"),
        top_rule="R5",
        current_lr=0.05,
        current_batch_size=128,
        expected=ExpectedDecision(
            event_types=frozenset({"architecture_change"}),
            cited_rule="R5",
            acceptable_remedy_directions=frozenset({"swap_activation"}),
            required_edit_op="swap_activation",
        ),
    ),
    Scenario(
        name="r4_capacity_add_block",
        description="R4 fires — clean gradients, train plateau, need more capacity",
        metrics_history=[
            {"epoch": 0, "train_acc": 0.70, "max_layer_grad_norm": 0.3, "dead_relu_fraction": 0.05},
            {"epoch": 1, "train_acc": 0.71, "max_layer_grad_norm": 0.3, "dead_relu_fraction": 0.05},
            {"epoch": 2, "train_acc": 0.71, "max_layer_grad_norm": 0.3, "dead_relu_fraction": 0.05},
        ],
        all_fired=_fired_only("R4"),
        top_rule="R4",
        current_lr=0.05,
        current_batch_size=128,
        expected=ExpectedDecision(
            event_types=frozenset({"architecture_change"}),
            cited_rule="R4",
            acceptable_remedy_directions=frozenset({"add_capacity"}),
            required_edit_op="add_block",
        ),
    ),
    Scenario(
        name="r3_early_stop",
        description="R3 fires — val loss flat for five epochs",
        metrics_history=[
            {"epoch": 0, "val_loss": 1.15, "train_loss": 0.80},
            {"epoch": 1, "val_loss": 1.15, "train_loss": 0.78},
            {"epoch": 2, "val_loss": 1.15, "train_loss": 0.76},
        ],
        all_fired=_fired_only("R3"),
        top_rule="R3",
        current_lr=0.05,
        current_batch_size=128,
        expected=ExpectedDecision(
            event_types=frozenset({"hyperparameter_change"}),
            cited_rule="R3",
            acceptable_remedy_directions=frozenset({"stop"}),
        ),
    ),
    Scenario(
        name="r1_high_update_ratio",
        description="R1 fires — update-to-param ratio too high",
        metrics_history=[
            {"epoch": 0, "update_to_param_ratio": 5e-2, "val_loss": 1.5},
            {"epoch": 1, "update_to_param_ratio": 6e-2, "val_loss": 1.8},
            {"epoch": 2, "update_to_param_ratio": 7e-2, "val_loss": 2.1},
        ],
        all_fired=_fired_only("R1"),
        top_rule="R1",
        current_lr=0.3,
        current_batch_size=128,
        expected=ExpectedDecision(
            event_types=frozenset({"hyperparameter_change"}),
            cited_rule="R1",
            acceptable_remedy_directions=frozenset({"decrease_lr"}),
        ),
    ),
    # Adversarial: precedence tie-break
    Scenario(
        name="r7_plus_r1_precedence_r7_wins",
        description="R7 and R1 both fire — stability beats tuning",
        metrics_history=[
            {"epoch": 0, "max_layer_grad_norm": 14.0, "update_to_param_ratio": 5e-2},
            {"epoch": 1, "max_layer_grad_norm": 14.2, "update_to_param_ratio": 5e-2},
            {"epoch": 2, "max_layer_grad_norm": 14.5, "update_to_param_ratio": 5e-2},
        ],
        all_fired={**_fired_only("R7"), "R1": True},
        top_rule="R7",  # the driver already picked the top per precedence
        current_lr=0.3,
        current_batch_size=128,
        expected=ExpectedDecision(
            event_types=frozenset({"hyperparameter_change"}),
            cited_rule="R7",
            acceptable_remedy_directions=frozenset({"decrease_lr"}),
        ),
    ),
]
