import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from env_rl.judge.deliverables import (  # noqa: E402
    HardFail,
    check_deliverables_exist,
    check_load_model_signature,
    check_run_config,
    load_submitted_model,
)


def _write_model_py(path: Path, *, signature: str = "def load_model():", body: str | None = None) -> None:
    body = body or (
        "    import torch\n"
        "    import torch.nn as nn\n"
        "    m = nn.Linear(4, 2)\n"
        "    m.load_state_dict(torch.load(str(__file__).rsplit('/', 1)[0] + '/best_model.pt'))\n"
        "    m.eval()\n"
        "    return m\n"
    )
    path.write_text(f"{signature}\n{body}")


def _write_valid_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    _write_model_py(workspace / "model.py")
    torch.save(nn.Linear(4, 2).state_dict(), workspace / "best_model.pt")
    (workspace / "run_config.json").write_text(
        json.dumps(
            {"seed": 42, "initial_hparams": {"lr": 0.1}, "max_epochs": 10}
        )
    )


def test_missing_model_py_hard_fails(tmp_path: Path) -> None:
    (tmp_path / "best_model.pt").write_bytes(b"\x00")
    (tmp_path / "run_config.json").write_text("{}")
    with pytest.raises(HardFail):
        check_deliverables_exist(tmp_path)


def test_missing_weights_hard_fails(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("def load_model(): return None\n")
    (tmp_path / "run_config.json").write_text("{}")
    with pytest.raises(HardFail):
        check_deliverables_exist(tmp_path)


def test_missing_run_config_hard_fails(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("def load_model(): return None\n")
    (tmp_path / "best_model.pt").write_bytes(b"\x00")
    with pytest.raises(HardFail):
        check_deliverables_exist(tmp_path)


def test_load_model_with_required_args_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    (tmp_path / "model.py").write_text(
        "def load_model(arg):\n    return None\n"
    )
    with pytest.raises(HardFail):
        check_load_model_signature(tmp_path)


def test_load_model_missing_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    (tmp_path / "model.py").write_text("x = 1\n")
    with pytest.raises(HardFail):
        check_load_model_signature(tmp_path)


def test_load_model_not_callable_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    (tmp_path / "model.py").write_text("load_model = 42\n")
    with pytest.raises(HardFail):
        check_load_model_signature(tmp_path)


def test_load_model_returns_nn_module_in_eval_mode(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    m = load_submitted_model(tmp_path)
    assert isinstance(m, nn.Module)
    assert not m.training


def test_load_model_in_training_mode_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    (tmp_path / "model.py").write_text(
        "import torch\nimport torch.nn as nn\n"
        "def load_model():\n"
        "    m = nn.Linear(4, 2)\n"
        "    m.load_state_dict(torch.load(str(__file__).rsplit('/', 1)[0] + '/best_model.pt'))\n"
        "    return m  # still in training mode\n"
    )
    with pytest.raises(HardFail):
        load_submitted_model(tmp_path)


def test_weight_shape_mismatch_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    # Save weights for a 4->2 net but model.py builds a 4->3 net
    torch.save(nn.Linear(4, 3).state_dict(), tmp_path / "best_model.pt")
    with pytest.raises(HardFail):
        load_submitted_model(tmp_path)


def test_run_config_missing_field_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"seed": 42, "max_epochs": 10})  # missing initial_hparams
    )
    with pytest.raises(HardFail):
        check_run_config(tmp_path)


def test_run_config_disagrees_with_session_start_hard_fails(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    logged = {
        "run_config": {
            "seed": 42,
            "initial_hparams": {"lr": 0.1},
            "max_epochs": 10,
        }
    }
    # Submitted file lies about seed
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {"seed": 99, "initial_hparams": {"lr": 0.1}, "max_epochs": 10}
        )
    )
    with pytest.raises(HardFail):
        check_run_config(tmp_path, session_start_record=logged)


def test_run_config_consistent_passes(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    logged = {
        "run_config": {
            "seed": 42,
            "initial_hparams": {"lr": 0.1},
            "max_epochs": 10,
        }
    }
    cfg = check_run_config(tmp_path, session_start_record=logged)
    assert cfg["seed"] == 42
