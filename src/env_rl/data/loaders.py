"""CIFAR-10 DataLoaders for the agent's workspace.

The test split is held out at the filesystem level but this module adds a
defense-in-depth guard: ``make_loader("test", ...)`` raises ``PermissionError``.
The judge uses :func:`make_test_loader_judge_only` instead, which is clearly
named and intentionally has no guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TestSplitAccessError(PermissionError):
    """Raised when workspace code attempts to load the held-out test split."""

    __test__ = False  # tell pytest this is not a test class


def make_loader(
    split: str,
    batch_size: int,
    data_dir: str | Path,
    *,
    num_workers: int = 2,
    shuffle: bool | None = None,
    manifest_path: str | Path | None = None,
) -> "DataLoader":
    """Build a CIFAR-10 DataLoader for the agent.

    ``split`` must be ``"train"`` or ``"val"``; requesting ``"test"`` raises
    ``TestSplitAccessError`` before any data path is touched.
    """
    if split == "test":
        raise TestSplitAccessError(
            "test split is held out; workspace code cannot load it. "
            "Evaluate only on 'val' during training."
        )
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    import json

    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    base = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=tf)

    manifest_path = Path(manifest_path) if manifest_path else Path(data_dir) / "splits.json"
    manifest = json.loads(Path(manifest_path).read_text())
    if split not in manifest:
        raise KeyError(f"manifest missing '{split}' key: {manifest_path}")
    indices = list(manifest[split])

    subset = Subset(base, indices)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def make_test_loader_judge_only(
    batch_size: int,
    data_dir: str | Path,
    *,
    num_workers: int = 2,
) -> "DataLoader":
    """JUDGE-SIDE ONLY. Load the full held-out test split.

    Any use of this function from agent code is a hard fail. The function is
    kept in a separate name so a workspace-side audit can grep for it.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    test = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=tf)
    return DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
