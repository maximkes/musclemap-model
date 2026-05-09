from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.dataset import MuscleActivationDataset, build_dataloaders, clean_label


def _write_seq(seq_dir: Path, *, T: int, n_muscles: int = 80, label: str) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    acts = np.zeros((T, n_muscles), dtype=np.float32)
    motion = np.zeros((T, 322), dtype=np.float32)
    np.save(seq_dir / "activations.npy", acts)
    np.save(seq_dir / "smplx_322.npy", motion)
    (seq_dir / "semantic_label.txt").write_text(label, encoding="utf-8")


def _base_config(dataset_root: Path, *, dataset_version: str = "", min_T: int = 30, max_T: int = 64) -> dict[str, Any]:
    return {
        "data": {
            "dataset_root": str(dataset_root),
            "dataset_version": dataset_version,
            "muscle_names_json": "muscle_names.json",
            "train_split": 0.90,
            "val_split": 0.05,
            "test_split": 0.05,
            "min_T": min_T,
            "max_T": max_T,
            "split_seed": 42,
        },
        "model": {"head": {"n_muscles": 80}},
        "training": {"batch_size": 2},
        "hardware": {"num_workers": 0, "pin_memory": False},
    }


@pytest.fixture()
def synthetic_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "muscle_names.json").write_text(json.dumps([f"m{i}" for i in range(80)]), encoding="utf-8")

    # 10 sequences, 3 canonical groups.
    # Group A: "walk"
    _write_seq(root / "walk_clip1", T=40, label="walk")
    _write_seq(root / "walk_clip2", T=60, label="walk")
    _write_seq(root / "walk_clip3", T=120, label="walk")
    _write_seq(root / "walk_clip4", T=10, label="walk")  # should be filtered by min_T
    # Group B: "run"
    _write_seq(root / "run_clip1", T=35, label="run")
    _write_seq(root / "run_clip2", T=80, label="run")
    _write_seq(root / "run_clip3", T=140, label="run")
    # Group C: numeric blob label -> fallback to clean_label(sample_id)
    _write_seq(root / "jump_clip1", T=50, label="0.12 0.34 0.56 0.78")
    _write_seq(root / "jump_clip2", T=55, label="1,2,3,4,5")
    _write_seq(root / "jump_clip3", T=70, label="0 1 2 3 4 5 6")
    return root


def test_split_has_no_canonical_overlap(synthetic_dataset_root: Path) -> None:
    config = _base_config(synthetic_dataset_root, min_T=30, max_T=64)

    train = MuscleActivationDataset(synthetic_dataset_root, config=config, split="train")
    val = MuscleActivationDataset(synthetic_dataset_root, config=config, split="val")
    test = MuscleActivationDataset(synthetic_dataset_root, config=config, split="test")

    train_labels = {x[3] for x in train._items}
    val_labels = {x[3] for x in val._items}
    test_labels = {x[3] for x in test._items}

    assert train_labels.isdisjoint(val_labels)
    assert train_labels.isdisjoint(test_labels)
    assert val_labels.isdisjoint(test_labels)
    assert (train_labels | val_labels | test_labels) <= {clean_label("walk"), clean_label("run"), clean_label("jump")}


def test_windowing_count_for_long_sequence(synthetic_dataset_root: Path) -> None:
    config = _base_config(synthetic_dataset_root, min_T=30, max_T=64)
    ds = MuscleActivationDataset(synthetic_dataset_root, config=config, split="train")

    # Find the long "walk" sequence with T=120 (should produce 3 windows at max_T=64, stride=32):
    # starts at 0, 32, 56 (last_start=56)
    windows = [it for it in ds._items if it[0].name == "walk_clip3"]
    assert len(windows) == 3
    assert [w[1] for w in windows] == [0, 32, 56]


def test_mask_padding_and_true_T(synthetic_dataset_root: Path) -> None:
    config = _base_config(synthetic_dataset_root, min_T=30, max_T=64)
    ds = MuscleActivationDataset(synthetic_dataset_root, config=config, split="train")

    # pick a short (unpadded beyond max_T) sequence: T=35 -> mask true for 35 then false.
    idx = next(i for i, it in enumerate(ds._items) if it[0].name == "run_clip1")
    item = ds[idx]
    assert item["acts"].shape == (64, 80)
    assert item["motion"].shape == (64, 322)
    assert item["mask"].shape == (64,)
    assert item["true_T"] == 35
    assert bool(item["mask"][:35].all()) is True
    assert bool((~item["mask"][35:]).all()) is True


def test_min_T_filter_removes_short_sequences(synthetic_dataset_root: Path) -> None:
    config = _base_config(synthetic_dataset_root, min_T=30, max_T=64)
    ds = MuscleActivationDataset(synthetic_dataset_root, config=config, split="train")
    names = {it[0].name for it in ds._items}
    assert "walk_clip4" not in names


def test_numeric_blob_label_falls_back_to_clean_label(synthetic_dataset_root: Path) -> None:
    config = _base_config(synthetic_dataset_root, min_T=30, max_T=64)
    ds = MuscleActivationDataset(synthetic_dataset_root, config=config, split="train")
    jump_items = [ds[i] for i, it in enumerate(ds._items) if it[0].name.startswith("jump_")]
    assert jump_items, "Expected jump_* items to be present"
    for ex in jump_items:
        assert ex["text"] == clean_label("jump")


def test_train_loader_non_empty_when_samples_fewer_than_batch(tmp_path: Path) -> None:
    """With drop_last=False, one train sample still yields one batch (regression guard)."""

    root = tmp_path / "data"
    root.mkdir(parents=True)
    (root / "muscle_names.json").write_text(json.dumps([f"m{i}" for i in range(80)]), encoding="utf-8")
    _write_seq(root / "single_seq", T=40, label="walk")

    config = _base_config(root, min_T=30, max_T=64)
    config["training"]["batch_size"] = 16
    train_dl, _, _ = build_dataloaders(config)
    assert len(train_dl.dataset) >= 1
    assert len(train_dl) >= 1


def test_dataset_version_subdir_used_when_set(tmp_path: Path) -> None:
    root = tmp_path / "data"
    v1 = root / "v1"
    v1.mkdir(parents=True, exist_ok=True)
    (v1 / "muscle_names.json").write_text(json.dumps([f"m{i}" for i in range(80)]), encoding="utf-8")
    _write_seq(v1 / "walk_clip1", T=40, label="walk")

    config = _base_config(root, dataset_version="v1", min_T=30, max_T=64)
    ds = MuscleActivationDataset(root, config=config, split="train")
    assert len(ds) == 1
    assert ds._items[0][0].parent == v1


