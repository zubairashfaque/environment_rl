"""The monitor — the only legitimate logging path in the environment.

Typical call sequence::

    monitor.start_session(run_config, monitor_config=cfg)
    monitor.attach(model)
    for epoch in range(max_epochs):
        for batch in train_loader:
            ...  # training step
            monitor.record_step()
        metrics = monitor.collect_epoch_metrics(model, val_loader, train_loss=..., lr=...)
        rule_evals = monitor.evaluate_rules(metrics)
        monitor.log_epoch(metrics)
        monitor.log_rule_eval(rule_evals)
        for rule, fired in rule_evals.items():
            if fired:
                monitor.log_decision("hyperparameter_change", cites=[rule], justification="...")
    monitor.end_session()
"""

from env_rl.monitor.session import (
    ALLOWED_EVENT_TYPES,
    SessionError,
    attach,
    collect_epoch_metrics,
    end_session,
    evaluate_rules,
    log_decision,
    log_epoch,
    log_rule_eval,
    record_step,
    start_session,
)

__all__ = [
    "ALLOWED_EVENT_TYPES",
    "SessionError",
    "attach",
    "collect_epoch_metrics",
    "end_session",
    "evaluate_rules",
    "log_decision",
    "log_epoch",
    "log_rule_eval",
    "record_step",
    "start_session",
]
