from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.dataset import MuscleActivationDataset


def _min_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "data": {
            "dataset_root": str(tmp_path),
            "dataset_version": "",
            "max_T": 16,
            "min_T": 1,
            "muscle_names_json": "muscle_names.json",
            "split_seed": 0,
            "train_split": 1.0,
            "val_split": 0.0,
            "test_split": 0.0,
        },
        "model": {"head": {"n_muscles": 80}},
        "training": {"batch_size": 2},
        "hardware": {"num_workers": 0, "pin_memory": False},
    }


def test_dataset_skips_truncated_npy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "muscle_names.json").write_text(str(["m"] * 80).replace("'", '"'), encoding="utf-8")

    # Build two "sequence dirs" so retry can hop.
    good = tmp_path / "seq_good"
    bad = tmp_path / "seq_bad"
    for d in (good, bad):
        d.mkdir(parents=True, exist_ok=True)
        (d / "semantic_label.txt").write_text("walk", encoding="utf-8")

    np.save(good / "activations.npy", np.zeros((16, 80), dtype=np.float32))
    np.save(good / "smplx_322.npy", np.zeros((16, 322), dtype=np.float32))
    np.save(bad / "activations.npy", np.zeros((16, 80), dtype=np.float32))
    np.save(bad / "smplx_322.npy", np.zeros((16, 322), dtype=np.float32))

    real_np_load = np.load

    bad_acts_calls = 0

    def fake_load(path: str | Path, *args: Any, **kwargs: Any):  # noqa: ANN001
        nonlocal bad_acts_calls
        p = Path(path)
        # Let dataset init read activations.npy once to get T,
        # then simulate a truncated file during __getitem__.
        if p.name == "activations.npy" and p.parent.name == "seq_bad":
            bad_acts_calls += 1
            if bad_acts_calls >= 2:
                raise EOFError("No data left in file")
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", fake_load)

    ds = MuscleActivationDataset(tmp_path, config=_min_config(tmp_path), split="train")

    # Force indexing the bad item; should retry and still return a valid sample.
    bad_idx = next(i for i, (seq_dir, *_rest) in enumerate(ds._items) if seq_dir.name == "seq_bad")  # type: ignore[attr-defined]
    sample = ds[bad_idx]
    assert sample["acts"].shape == (16, 80)
    assert sample["motion"].shape == (16, 322)
