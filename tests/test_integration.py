from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from app.app import predict_and_save as app_predict_and_save
from app.demo import predict_and_save as demo_predict_and_save


class DummyModel(nn.Module):
    def __init__(self, *, T: int = 12) -> None:
        super().__init__()
        self.T = T

    def forward(self, text_tokens: Any, motion_tokens: Any | None = None) -> tuple[torch.Tensor, torch.Tensor, Any]:
        _ = (motion_tokens,)
        B = len(text_tokens)
        logits = torch.linspace(-2.0, 2.0, steps=self.T).view(1, self.T, 1).repeat(B, 1, 80)
        pred_log_T = torch.log(torch.full((B, 1), float(self.T), dtype=torch.float32))
        return logits, pred_log_T, {}


def test_app_predict_writes_npy_float32_and_sigmoid_range(tmp_path: Path) -> None:
    model = DummyModel(T=10)
    muscle_names = [f"m{i}" for i in range(80)]
    out_dir = tmp_path / "out"

    out_path, _png, acts = app_predict_and_save(prompt="a person walks", model=model, muscle_names=muscle_names, out_dir=out_dir)
    assert out_path.exists()

    loaded = np.load(out_path)
    assert loaded.dtype == np.float32
    assert loaded.shape == (10, 80)
    assert np.all(loaded >= 0.0) and np.all(loaded <= 1.0)
    assert acts.shape == (10, 80)


def test_demo_predict_writes_expected_file(tmp_path: Path) -> None:
    model = DummyModel(T=7)
    out_path = tmp_path / "acts.npy"
    cfg = {"data": {"dataset_root": str(tmp_path)}}

    acts = demo_predict_and_save(text="run", output_npy=out_path, config=cfg, model=model)
    loaded = np.load(out_path)
    assert loaded.dtype == np.float32
    assert loaded.shape == (7, 80)
    assert np.allclose(loaded, acts)

