from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.trainer import Trainer


class TinyDataset(Dataset[dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, Any]] = []
        for _ in range(4):
            self.items.append(
                {
                    "text": "walk",
                    "acts": torch.zeros((16, 80), dtype=torch.float32),
                    "mask": torch.ones((16,), dtype=torch.bool),
                    "true_T": 16,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    acts = torch.stack([b["acts"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    true_T = torch.tensor([b["true_T"] for b in batch], dtype=torch.int64)
    text = [b["text"] for b in batch]
    return {"acts": acts, "mask": mask, "true_T": true_T, "text": text}


class DummyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation_head = nn.Linear(1, 1, bias=False)
        self.w = nn.Parameter(torch.tensor(0.0))

    def forward(self, text_tokens: Any, motion_tokens: Any | None = None) -> tuple[torch.Tensor, torch.Tensor, Any]:
        _ = (text_tokens, motion_tokens)
        B = len(text_tokens)
        T, N = 16, 80
        logits = self.w * torch.ones((B, T, N), dtype=torch.float32)
        pred_log_T = torch.log(torch.full((B, 1), float(T), dtype=torch.float32))
        return logits, pred_log_T, {}

    def parameters_to_train(self) -> list[nn.Parameter]:
        return [self.w]

    def apply_lora(self, _config: dict[str, Any]) -> None:
        # Add a trainable parameter that looks like LoRA.
        self.lora_A = nn.Parameter(torch.ones(1))  # type: ignore[attr-defined]
        self.lora_A.requires_grad = True  # type: ignore[attr-defined]


def _base_config(checkpoint_dir: Path) -> dict[str, Any]:
    return {
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "accumulation_steps": 1,
            "learning_rate": 1e-1,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "gradient_clip_norm": 1.0,
            "mixed_precision": "",
            "unfreeze_after_epoch": 1,
            "lora_lr": 5e-6,
            "loss": {"bce_weight": 1.0, "smoothness_weight": 0.0, "length_weight": 0.0},
        },
        "hardware": {"find_unused_parameters": False},
        "logging": {"checkpoint_dir": str(checkpoint_dir), "keep_last_n_checkpoints": 3, "eval_every_n_epochs": 1},
    }


def test_train_epoch_cpu_runs_and_checkpoint_restores_params(tmp_path: Path) -> None:
    ds = TinyDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=_collate)
    model = DummyTrainModel()
    cfg = _base_config(tmp_path / "ckpt")

    trainer = Trainer(config=cfg, model=model, train_loader=dl, val_loader=dl, device=torch.device("cpu"))
    stats = trainer.train_epoch()
    assert "loss" in stats

    before = float(model.w.detach().cpu())
    trainer.save_checkpoint(epoch=0, val_loss=stats["loss"])

    # Modify param, then load and ensure it is restored.
    with torch.no_grad():
        model.w.add_(10.0)
    assert float(model.w.detach().cpu()) != before

    start_epoch = trainer.load_checkpoint()
    assert start_epoch == 1
    assert float(model.w.detach().cpu()) == before


def test_stage2_transition_adds_lora_params_to_optimizer(tmp_path: Path) -> None:
    ds = TinyDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=_collate)
    model = DummyTrainModel()
    cfg = _base_config(tmp_path / "ckpt")

    trainer = Trainer(config=cfg, model=model, train_loader=dl, val_loader=dl, device=torch.device("cpu"))
    assert not any("lora" in n for n, _p in model.named_parameters())

    transitioned = trainer.maybe_transition_stage2(epoch=1)
    assert transitioned is True
    assert any("lora" in n for n, _p in model.named_parameters())
    assert any(abs(pg.get("lr", 0.0) - cfg["training"]["lora_lr"]) < 1e-12 for pg in trainer.optimizer.param_groups)


def test_load_checkpoint_skips_incompatible_optimizer(tmp_path: Path) -> None:
    """Resume weights even when saved optimizer param_groups do not match (e.g. LoRA vs stage1)."""

    ds = TinyDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=_collate)
    model = DummyTrainModel()
    cfg = _base_config(tmp_path / "ckpt")
    trainer = Trainer(config=cfg, model=model, train_loader=dl, val_loader=dl, device=torch.device("cpu"))
    trainer.train_epoch()
    expected_w = float(model.w.detach().cpu())
    trainer.save_checkpoint(epoch=0, val_loss=0.1)

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    latest = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
    blob = torch.load(latest, map_location="cpu")
    decoy = nn.Linear(2, 2)
    extra = nn.Parameter(torch.zeros(1))
    opt_two_groups = AdamW(
        [{"params": decoy.parameters(), "lr": 1e-3}, {"params": [extra], "lr": 5e-6}],
    )
    blob["optimizer"] = opt_two_groups.state_dict()
    torch.save(blob, latest)

    model2 = DummyTrainModel()
    with torch.no_grad():
        model2.w.fill_(99.0)
    trainer2 = Trainer(config=cfg, model=model2, train_loader=dl, val_loader=dl, device=torch.device("cpu"))
    start = trainer2.load_checkpoint()
    assert start == 1
    assert abs(float(model2.w.detach().cpu()) - expected_w) < 1e-5


def test_fit_skips_loop_when_checkpoint_epoch_reaches_training_epochs(tmp_path: Path, caplog: Any) -> None:
    """Resuming from the last saved epoch with training.epochs equal to that run yields no steps."""

    ds = TinyDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=_collate)
    model = DummyTrainModel()
    cfg = _base_config(tmp_path / "ckpt")
    cfg["training"]["epochs"] = 5
    trainer0 = Trainer(config=cfg, model=model, train_loader=dl, val_loader=dl, device=torch.device("cpu"))
    ckpt_path = tmp_path / "ckpt" / "epoch_0004.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 4,
            "model": model.state_dict(),
            "optimizer": trainer0.optimizer.state_dict(),
            "scheduler": trainer0.scheduler.state_dict(),
        },
        ckpt_path,
    )

    model2 = DummyTrainModel()
    trainer1 = Trainer(config=cfg, model=model2, train_loader=dl, val_loader=dl, device=torch.device("cpu"))
    with caplog.at_level(logging.WARNING, logger="src.trainer"):
        trainer1.fit()
    assert "No epochs to run" in caplog.text

