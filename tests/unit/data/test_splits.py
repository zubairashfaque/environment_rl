import json
from pathlib import Path

import pytest

from env_rl.data.splits import TRAINVAL_SIZE, build_splits, main


def test_build_splits_deterministic_under_same_seed() -> None:
    a = build_splits(seed=42)
    b = build_splits(seed=42)
    assert a == b


def test_build_splits_different_seeds_produce_different_splits() -> None:
    a = build_splits(seed=42)
    b = build_splits(seed=7)
    assert a["train"] != b["train"]
    assert a["val"] != b["val"]


def test_train_and_val_disjoint_and_cover_trainval_range() -> None:
    m = build_splits(seed=42)
    train = set(m["train"])
    val = set(m["val"])
    assert train.isdisjoint(val)
    assert train.union(val) == set(range(TRAINVAL_SIZE))


def test_val_size_matches_fraction() -> None:
    m = build_splits(seed=42, val_frac=0.1)
    assert len(m["val"]) == 5_000
    assert len(m["train"]) == 45_000


def test_manifest_contains_test_source_marker() -> None:
    m = build_splits(seed=42)
    assert m["test_source"] == "torchvision_test"
    assert "test" not in m  # no test-index list on the agent-facing manifest


def test_val_frac_bounds_enforced() -> None:
    with pytest.raises(ValueError):
        build_splits(seed=42, val_frac=0.0)
    with pytest.raises(ValueError):
        build_splits(seed=42, val_frac=1.0)


def test_cli_writes_manifest_to_disk(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "splits.json"
    monkeypatch.setattr("sys.argv", ["splits", "--seed", "42", "--out", str(out)])
    main()
    assert out.exists()
    m = json.loads(out.read_text())
    assert m["seed"] == 42
    assert set(m["train"]).isdisjoint(set(m["val"]))
