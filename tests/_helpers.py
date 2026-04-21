def build_history(**signals) -> list[dict]:
    """Build a metrics history from parallel per-epoch signals.

    Example: build_history(val_loss=[1.0, 0.9], train_acc=[0.3, 0.5])
    -> [{"val_loss": 1.0, "train_acc": 0.3}, {"val_loss": 0.9, "train_acc": 0.5}]
    """
    lengths = {len(v) for v in signals.values()}
    assert len(lengths) == 1, "all signals must have same length"
    n = lengths.pop()
    return [{k: signals[k][i] for k in signals} for i in range(n)]
