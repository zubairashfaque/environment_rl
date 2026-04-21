import json
from pathlib import Path

import pytest

from env_rl import monitor
from env_rl.monitor import session as _session_mod


@pytest.fixture(autouse=True)
def _clean_session():
    _session_mod._reset_for_tests()
    yield
    _session_mod._reset_for_tests()


@pytest.fixture
def mcfg(tmp_log_dir: Path, monitor_config: dict) -> dict:
    cfg = dict(monitor_config)
    cfg["log_dir"] = str(tmp_log_dir)
    cfg["root_hash"] = "0" * 64
    return cfg


def _run_config() -> dict:
    return {"seed": 42, "max_epochs": 10, "lr": 0.1, "batch_size": 128}


def test_start_session_creates_three_logs_with_bookend(
    tmp_log_dir: Path, mcfg: dict
) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    for name in ("metrics_log.jsonl", "decision_log.jsonl", "rule_evaluations.jsonl"):
        path = tmp_log_dir / name
        assert path.exists()
        first = json.loads(path.read_text().splitlines()[0])
        assert first["payload"]["kind"] == "session_start"
        assert first["payload"]["run_config"] == _run_config()


def test_start_session_rejects_duplicate_call(mcfg: dict) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    with pytest.raises(monitor.SessionError):
        monitor.start_session(_run_config(), monitor_config=mcfg)


def test_attach_before_start_session_raises(mcfg: dict) -> None:
    import torch.nn as nn

    with pytest.raises(monitor.SessionError):
        monitor.attach(nn.Linear(4, 2))


def test_log_decision_without_fired_rule_raises(mcfg: dict) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    metrics = monitor.collect_epoch_metrics(model=None)
    evals = monitor.evaluate_rules(metrics)
    monitor.log_epoch(metrics)
    monitor.log_rule_eval(evals)
    # No rule actually fired on an empty history; citing R7 must raise.
    with pytest.raises(ValueError):
        monitor.log_decision("hyperparameter_change", cites=["R7"], justification="x")


def test_log_decision_accepts_rule_fired_in_window(mcfg: dict) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    # Force a rule to have fired at the current epoch by logging a rule_eval
    # with R7=True. The decision at the same epoch citing R7 must succeed.
    metrics = monitor.collect_epoch_metrics(model=None, val_loss=1.0)
    monitor.log_epoch(metrics)
    monitor.log_rule_eval({f"R{i}": (i == 7) for i in range(1, 8)})
    monitor.log_decision(
        "hyperparameter_change", cites=["R7"], justification="dropped LR"
    )


def test_log_decision_bad_event_type_raises(mcfg: dict) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    with pytest.raises(ValueError):
        monitor.log_decision("wild_guess", cites=[], justification="")


def test_end_session_writes_bookends_and_closes(
    tmp_log_dir: Path, mcfg: dict
) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    monitor.end_session()
    for name in ("metrics_log.jsonl", "decision_log.jsonl", "rule_evaluations.jsonl"):
        path = tmp_log_dir / name
        lines = path.read_text().splitlines()
        last = json.loads(lines[-1])
        assert last["payload"]["kind"] == "session_end"


def test_end_session_without_active_session_raises() -> None:
    with pytest.raises(monitor.SessionError):
        monitor.end_session()


def test_exceptions_surface_through_monitor(mcfg: dict) -> None:
    # A bad value should raise, not be silently swallowed.
    monitor.start_session(_run_config(), monitor_config=mcfg)
    with pytest.raises(ValueError):
        monitor.log_decision("bogus_event", cites=[], justification="")
    # Session remains intact so the caller can decide to abort.
    monitor.end_session()


def test_epoch_number_monotonic(mcfg: dict) -> None:
    monitor.start_session(_run_config(), monitor_config=mcfg)
    m0 = monitor.collect_epoch_metrics(model=None)
    m1 = monitor.collect_epoch_metrics(model=None)
    m2 = monitor.collect_epoch_metrics(model=None)
    assert [m0["epoch"], m1["epoch"], m2["epoch"]] == [0, 1, 2]
