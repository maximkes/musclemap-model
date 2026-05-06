from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def clean_label(sample_id: str) -> str:
    """Convert a sequence id into a canonical action label."""

    s = re.sub(r"_clip\d+$", "", sample_id, flags=re.IGNORECASE)
    s = re.sub(r"_\d+$", "", s)
    return s.replace("_", " ").strip().lower()


def _is_numeric_blob(text: str) -> bool:
    """Return True when text looks like a numeric blob."""

    tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    if not tokens:
        return True
    numeric = 0
    for t in tokens:
        try:
            float(t)
        except ValueError:
            continue
        else:
            numeric += 1
    return (numeric / len(tokens)) > 0.5


def _load_muscle_names(muscle_names_path: Path, n_muscles: int) -> list[str]:
    with muscle_names_path.open("r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or any(not isinstance(x, str) for x in names):
        raise ValueError(f"Invalid muscle_names.json format at {muscle_names_path}")
    if len(names) != n_muscles:
        raise ValueError(f"Expected {n_muscles} muscles, got {len(names)} at {muscle_names_path}")
    return names


def _select_dataset_root(dataset_root: Path, dataset_version: str) -> Path:
    if not dataset_version:
        return dataset_root
    versioned = dataset_root / dataset_version
    if versioned.exists():
        return versioned
    logger.warning("dataset_version=%s missing; falling back to %s", dataset_version, dataset_root)
    return dataset_root


def _scan_sequence_dirs(dataset_root: Path) -> list[Path]:
    """Find all sequence directories under dataset_root."""

    seq_dirs: set[Path] = set()
    for act_path in dataset_root.rglob("activations.npy"):
        seq_dir = act_path.parent
        if (seq_dir / "smplx_322.npy").exists() and (seq_dir / "semantic_label.txt").exists():
            seq_dirs.add(seq_dir)
    return sorted(seq_dirs)


class MuscleActivationDataset(Dataset[dict[str, Any]]):
    """Dataset returning windowed/padded sequences for training and evaluation."""

    def __init__(self, dataset_root: Path, config: dict[str, Any], split: str) -> None:
        self.config = config
        data_cfg = config["data"]
        model_cfg = config["model"]["head"]

        self.dataset_root = _select_dataset_root(dataset_root, str(data_cfg.get("dataset_version", "")))
        self.n_muscles = int(model_cfg["n_muscles"])
        self.max_T = int(data_cfg["max_T"])
        self.min_T = int(data_cfg.get("min_T", 30))
        self.split = split

        muscle_names_json = str(data_cfg.get("muscle_names_json", "muscle_names.json"))
        self.muscle_names = _load_muscle_names(self.dataset_root / muscle_names_json, self.n_muscles)

        seq_dirs = _scan_sequence_dirs(self.dataset_root)
        seq_infos: list[tuple[Path, int, str]] = []
        # (seq_dir, T, canonical)
        for seq_dir in seq_dirs:
            sample_id = seq_dir.name
            acts_np = np.load(seq_dir / "activations.npy")
            if acts_np.ndim != 2:
                raise ValueError(f"activations.npy must be 2D, got {acts_np.shape} in {seq_dir}")
            T = int(acts_np.shape[0])
            if T < self.min_T:
                continue

            label_text = (seq_dir / "semantic_label.txt").read_text(encoding="utf-8").strip()
            if not label_text or _is_numeric_blob(label_text):
                text = clean_label(sample_id)
            else:
                text = label_text.strip().lower()
            canonical = clean_label(text.replace(" ", "_"))
            seq_infos.append((seq_dir, T, canonical))

        groups: dict[str, list[tuple[Path, int]]] = {}
        for seq_dir, T, canonical in seq_infos:
            groups.setdefault(canonical, []).append((seq_dir, T))

        split_seed = int(data_cfg.get("split_seed", 42))
        rng = random.Random(split_seed)
        canonical_names = sorted(groups.keys())
        rng.shuffle(canonical_names)

        train_p = float(data_cfg.get("train_split", 0.90))
        val_p = float(data_cfg.get("val_split", 0.05))
        test_p = float(data_cfg.get("test_split", 0.05))
        if not np.isclose(train_p + val_p + test_p, 1.0):
            raise ValueError("train/val/test splits must sum to 1.0")

        n_groups = len(canonical_names)
        n_train = int(round(train_p * n_groups))
        n_val = int(round(val_p * n_groups))
        n_train = min(n_train, n_groups)
        n_val = min(n_val, n_groups - n_train)

        train_groups = set(canonical_names[:n_train])
        val_groups = set(canonical_names[n_train : n_train + n_val])
        test_groups = set(canonical_names[n_train + n_val :])

        if split == "train":
            keep = train_groups
        elif split == "val":
            keep = val_groups
        elif split == "test":
            keep = test_groups
        else:
            raise ValueError(f"Unknown split: {split}")

        stride = max(1, self.max_T // 2)
        self._items: list[tuple[Path, int, int, str]] = []
        # (seq_dir, start, true_T, text)
        for canonical in sorted(keep):
            for seq_dir, T in groups[canonical]:
                if T <= self.max_T:
                    self._items.append((seq_dir, 0, T, canonical))
                    continue
                last_start = T - self.max_T
                start = 0
                while start < last_start:
                    self._items.append((seq_dir, start, self.max_T, canonical))
                    start += stride
                self._items.append((seq_dir, last_start, self.max_T, canonical))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_dir, start, true_T, text = self._items[idx]
        acts_np = np.load(seq_dir / "activations.npy")
        motion_np = np.load(seq_dir / "smplx_322.npy")

        if acts_np.shape[1] != self.n_muscles:
            raise ValueError(f"Expected {self.n_muscles} muscles, got {acts_np.shape[1]} in {seq_dir}")
        if motion_np.ndim != 2 or motion_np.shape[1] != 322:
            raise ValueError(f"smplx_322.npy must be [T,322], got {motion_np.shape} in {seq_dir}")
        if acts_np.shape[0] != motion_np.shape[0]:
            raise ValueError(f"T mismatch between acts and motion in {seq_dir}")

        acts_win = acts_np[start : start + self.max_T]
        motion_win = motion_np[start : start + self.max_T]
        T_win = int(acts_win.shape[0])

        acts = torch.zeros((self.max_T, self.n_muscles), dtype=torch.float32)
        motion = torch.zeros((self.max_T, 322), dtype=torch.float32)
        mask = torch.zeros((self.max_T,), dtype=torch.bool)

        acts[:T_win] = torch.from_numpy(acts_win.astype(np.float32, copy=False))
        motion[:T_win] = torch.from_numpy(motion_win.astype(np.float32, copy=False))
        mask[:T_win] = True

        return {
            "text": text,
            "motion": motion,
            "acts": acts,
            "mask": mask,
            "true_T": int(true_T),
        }


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Build train/val/test dataloaders."""

    data_cfg = config["data"]
    dataset_root = Path(str(data_cfg["dataset_root"]))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["hardware"].get("num_workers", 0))
    pin_memory = bool(config["hardware"].get("pin_memory", False))

    train_ds = MuscleActivationDataset(dataset_root, config=config, split="train")
    val_ds = MuscleActivationDataset(dataset_root, config=config, split="val")
    test_ds = MuscleActivationDataset(dataset_root, config=config, split="test")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    eval_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    val_dl = DataLoader(val_ds, **eval_kwargs)
    test_dl = DataLoader(test_ds, **eval_kwargs)
    return train_dl, val_dl, test_dl

