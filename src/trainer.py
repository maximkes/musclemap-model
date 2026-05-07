from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

from contextlib import nullcontext
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.losses import activation_loss

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    """Return True for rank 0 (or non-distributed)."""

    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    """Atomically write a torch checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _linear_warmup_cosine_decay(warmup_steps: int, total_steps: int) -> Any:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    return lr_lambda


def _autocast_ctx(device_type: str, enabled: bool) -> Any:
    """Return an autocast context compatible across torch versions."""

    if not enabled:
        return nullcontext()
    try:
        from torch.amp import autocast as amp_autocast  # type: ignore[attr-defined]

        return amp_autocast(device_type=device_type, dtype=torch.bfloat16)
    except Exception:  # noqa: BLE001
        from torch.cuda.amp import autocast as cuda_autocast

        return cuda_autocast(dtype=torch.bfloat16)


@dataclass
class Trainer:
    """Training loop wrapper."""

    config: dict[str, Any]
    model: nn.Module
    train_loader: DataLoader[dict[str, Any]]
    val_loader: DataLoader[dict[str, Any]] | None = None
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # DDP wraps activation_head only (if distributed initialized).
        self.activation_head = getattr(self.model, "activation_head", None)
        if self.activation_head is None:
            raise AttributeError("model must have activation_head attribute")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.activation_head = torch.nn.parallel.DistributedDataParallel(  # type: ignore[assignment]
                self.activation_head,
                device_ids=[local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=bool(self.config.get("hardware", {}).get("find_unused_parameters", False)),
            )
            self.model.activation_head = self.activation_head  # type: ignore[attr-defined]

        lr = float(self.config["training"]["learning_rate"])
        wd = float(self.config["training"].get("weight_decay", 0.0))
        self.optimizer = AdamW(self.model.parameters_to_train(), lr=lr, weight_decay=wd)  # type: ignore[attr-defined]

        epochs = int(self.config["training"]["epochs"])
        steps_per_epoch = max(1, len(self.train_loader))
        accum = int(self.config["training"].get("accumulation_steps", 1))
        total_steps = max(1, (epochs * steps_per_epoch) // max(1, accum))
        warmup_steps = int(self.config["training"].get("warmup_steps", 0))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_linear_warmup_cosine_decay(warmup_steps, total_steps))

        self._global_step = 0

    def train_epoch(self) -> dict[str, Any]:
        """Run one training epoch."""

        self.model.train()
        accum_steps = int(self.config["training"].get("accumulation_steps", 1))
        clip_norm = float(self.config["training"].get("gradient_clip_norm", 1.0))
        mp = str(self.config["training"].get("mixed_precision", "")).lower()
        use_autocast = (mp == "bf16") and (self.device.type == "cuda")

        total_loss = 0.0
        n_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            acts = batch["acts"].to(self.device)           # [B, T, 80]
            mask = batch["mask"].to(self.device)           # [B, T]
            true_T = torch.as_tensor(batch["true_T"], device=self.device)
            text_tokens = batch.get("text_tokens", batch["text"])
            motion_tokens = batch.get("motion_tokens", None)

            # MotionGPT backbone requires a "length" key in the batch dict.
            # Use true_T if available (actual per-sample lengths), otherwise
            # fall back to acts sequence length as a safe default.
            lengths = batch.get("lengths", true_T.tolist() if true_T.ndim > 0 else [acts.shape[1]] * acts.shape[0])

            with _autocast_ctx(self.device.type, enabled=use_autocast):
                logits, pred_log_T, _ = self.model(
                    text_tokens,
                    motion_tokens=motion_tokens,
                    lengths=lengths,          # passed through to backbone batch dict
                )
                loss_cfg = dict(self.config)
                loss_cfg["pred_log_T"] = pred_log_T
                loss_cfg["true_T"] = true_T
                loss, _parts = activation_loss(logits, acts, mask, loss_cfg)
                loss = loss / float(accum_steps)

            loss.backward()

            is_accum_step = ((step + 1) % accum_steps) == 0
            is_last_step = (step + 1) == len(self.train_loader)

            if is_accum_step or is_last_step:
                clip_grad_norm_(self.model.parameters_to_train(), max_norm=clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._global_step += 1

            total_loss += float(loss.detach()) * float(accum_steps)
            n_steps += 1

        return {"loss": total_loss / max(1, n_steps)}

    def val_epoch(self) -> dict[str, Any]:
        """Run one validation epoch."""

        if self.val_loader is None:
            return {"loss": float("nan")}
        self.model.eval()
        total_loss = 0.0
        n_steps = 0
        with torch.no_grad():
            for batch in self.val_loader:
                acts = batch["acts"].to(self.device)
                mask = batch["mask"].to(self.device)
                true_T = torch.as_tensor(batch["true_T"], device=self.device)
                text_tokens = batch.get("text_tokens", batch["text"])
                motion_tokens = batch.get("motion_tokens", None)

                logits, pred_log_T, _ = self.model(text_tokens, motion_tokens=motion_tokens)
                loss_cfg = dict(self.config)
                loss_cfg["pred_log_T"] = pred_log_T
                loss_cfg["true_T"] = true_T
                loss, _ = activation_loss(logits, acts, mask, loss_cfg)
                total_loss += float(loss.detach().cpu())
                n_steps += 1
        return {"loss": total_loss / max(1, n_steps)}

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save checkpoint to disk."""

        ckpt_dir = Path(str(self.config["logging"]["checkpoint_dir"]))
        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        if not _is_rank0():
            return
        state = {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        _atomic_torch_save(state, ckpt_path)

        keep_n = int(self.config["logging"].get("keep_last_n_checkpoints", 3))
        ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
        if keep_n > 0 and len(ckpts) > keep_n:
            for p in ckpts[: -keep_n]:
                try:
                    p.unlink()
                except OSError:
                    logger.warning("Failed to delete old checkpoint %s", p)

    def load_checkpoint(self) -> int:
        """Load checkpoint and return start epoch."""

        ckpt_dir = Path(str(self.config["logging"]["checkpoint_dir"]))
        ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
        if not ckpts:
            return 0
        latest = ckpts[-1]
        state = torch.load(latest, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        return int(state.get("epoch", 0)) + 1

    def maybe_transition_stage2(self, epoch: int) -> bool:
        """Apply LoRA and update optimizer if entering stage 2."""

        unfreeze_after = int(self.config["training"].get("unfreeze_after_epoch", 10_000))
        if epoch != unfreeze_after:
            return False

        self.save_checkpoint(epoch=epoch, val_loss=float("nan"))
        self.model.apply_lora(self.config)  # type: ignore[attr-defined]

        lora_lr = float(self.config["training"].get("lora_lr", 5e-6))
        lora_params: list[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and ("lora" in name.lower()):
                lora_params.append(p)
        if lora_params:
            self.optimizer.add_param_group({"params": lora_params, "lr": lora_lr})
        return True

    def fit(self) -> None:
        """Run the full training loop."""

        start_epoch = self.load_checkpoint()
        epochs = int(self.config["training"]["epochs"])

        for epoch in range(start_epoch, epochs):
            self.maybe_transition_stage2(epoch)
            train_stats = self.train_epoch()
            val_stats = self.val_epoch()

            if _is_rank0():
                logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_stats["loss"], val_stats["loss"])
            if (epoch + 1) % int(self.config["logging"].get("eval_every_n_epochs", 1)) == 0:
                self.save_checkpoint(epoch=epoch, val_loss=float(val_stats["loss"]))

