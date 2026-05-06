from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def tiny_config() -> dict[str, Any]:
    """Return a minimal config dict for unit tests."""

    return {"data": {"dataset_root": "dummy"}}


@pytest.fixture()
def tiny_dataset_dir(tmp_path: Path) -> Path:
    """Create a tiny fake dataset dir (stub)."""

    root = tmp_path / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def synthetic_batch() -> dict[str, Any]:
    """Return a small synthetic batch for unit tests."""

    B, T, N = 2, 16, 80
    return {
        "targets": torch.zeros((B, T, N), dtype=torch.float32),
        "mask": torch.ones((B, T), dtype=torch.bool),
        "pred": np.zeros((B, T, N), dtype=np.float32),
        "true": np.zeros((B, T, N), dtype=np.float32),
    }

