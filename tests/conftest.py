from pathlib import Path

import pytest


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "judge_logs"
    d.mkdir()
    return d


@pytest.fixture
def root_hash() -> str:
    return "0" * 64
