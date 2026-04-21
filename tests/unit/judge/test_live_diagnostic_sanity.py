import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.judge.deliverables import HardFail  # noqa: E402
from env_rl.judge.live_diag import (  # noqa: E402
    compare_norms_within_tolerance,
    measure_live_gradient_norms,
    run_live_diagnostic,
)


def _toy_batch(in_dim: int = 4, n: int = 4, classes: int = 2):
    torch.manual_seed(0)
    return (torch.randn(n, in_dim), torch.randint(0, classes, (n,)))


def test_measure_live_gradient_norms_nonzero_after_fwd_bwd() -> None:
    model = nn.Linear(4, 2)
    norms = measure_live_gradient_norms(model, _toy_batch())
    assert all(v >= 0 for v in norms.values())
    assert sum(norms.values()) > 0


def test_compare_norms_within_tolerance_passes_close_match() -> None:
    live = {"a": 2.8, "b": 1.0}
    logged = {"a": 2.7, "b": 1.05}
    compare_norms_within_tolerance(live, logged, tolerance=0.30)


def test_compare_norms_within_tolerance_hard_fails_gross_mismatch() -> None:
    live = {"a": 120.0}
    logged = {"a": 2.8}
    with pytest.raises(HardFail):
        compare_norms_within_tolerance(live, logged, tolerance=0.30)


def test_compare_norms_hard_fails_on_empty_live() -> None:
    with pytest.raises(HardFail):
        compare_norms_within_tolerance({}, {"a": 1.0}, tolerance=0.30)


def test_compare_norms_hard_fails_on_missing_logged_layer() -> None:
    with pytest.raises(HardFail):
        compare_norms_within_tolerance({"a": 1.0}, {}, tolerance=0.30)


def test_run_live_diagnostic_passes_when_logged_matches_live() -> None:
    model = nn.Linear(4, 2)
    batches = [_toy_batch() for _ in range(2)]
    # compute once to get ~correct logged value
    norms = measure_live_gradient_norms(model, batches[0])
    logged = {"per_layer_grad_norm": dict(norms)}
    live = run_live_diagnostic(model, batches, logged, tolerance=0.50, max_batches=2)
    assert live  # dict of computed averages


def test_run_live_diagnostic_hard_fails_on_fabricated_trajectory() -> None:
    model = nn.Linear(4, 2)
    batches = [_toy_batch() for _ in range(2)]
    # Logged says tiny gradients; live will be normal → mismatch
    logged = {"per_layer_grad_norm": {"weight": 1e-6, "bias": 1e-6}}
    with pytest.raises(HardFail):
        run_live_diagnostic(model, batches, logged, tolerance=0.30, max_batches=2)
