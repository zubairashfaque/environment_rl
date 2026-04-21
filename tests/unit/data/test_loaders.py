import pytest

from env_rl.data.loaders import TestSplitAccessError, make_loader


def test_make_loader_rejects_test_split() -> None:
    with pytest.raises(TestSplitAccessError):
        make_loader(split="test", batch_size=32, data_dir="./nonexistent")


def test_make_loader_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        make_loader(split="banana", batch_size=32, data_dir="./nonexistent")


def test_test_split_access_error_is_permission_error() -> None:
    assert issubclass(TestSplitAccessError, PermissionError)
