from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def activation_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute activation loss terms.

    Args:
        logits: Raw logits [B, T, N]
        targets: Binary targets (float) [B, T, N]
        mask: Valid-frame mask [B, T] (bool)
        config: Training config dict (expects weights under training.loss.*)
    """

    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have same shape, got {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if mask.ndim != 2 or mask.shape[0] != logits.shape[0] or mask.shape[1] != logits.shape[1]:
        raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")

    loss_cfg = config.get("training", {}).get("loss", {})
    bce_w = float(loss_cfg.get("bce_weight", 1.0))
    smooth_w = float(loss_cfg.get("smoothness_weight", 0.1))
    length_w = float(loss_cfg.get("length_weight", 0.1))

    valid = mask.to(dtype=logits.dtype).unsqueeze(-1)  # [B, T, 1]
    denom = valid.sum().clamp_min(1.0) * logits.shape[-1]

    bce_raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B, T, N]
    bce = (bce_raw * valid).sum() / denom

    probs = torch.sigmoid(logits)  # [B, T, N]
    if logits.shape[1] >= 2:
        pair_valid = (mask[:, 1:] & mask[:, :-1]).to(dtype=logits.dtype).unsqueeze(-1)  # [B, T-1, 1]
        smooth = ((probs[:, 1:] - probs[:, :-1]).abs() * pair_valid).sum() / (
            pair_valid.sum().clamp_min(1.0) * logits.shape[-1]
        )
    else:
        smooth = logits.new_zeros(())

    length = logits.new_zeros(())
    pred_log_T = config.get("pred_log_T", None)
    true_T = config.get("true_T", None)
    true_log_T = config.get("true_log_T", None)
    if isinstance(pred_log_T, Tensor) and pred_log_T.numel() > 0:
        if isinstance(true_log_T, Tensor) and true_log_T.shape == pred_log_T.shape:
            length = nn.SmoothL1Loss()(pred_log_T, true_log_T)
        elif isinstance(true_T, Tensor):
            t = true_T.to(dtype=pred_log_T.dtype).clamp_min(1.0).view(-1, 1)
            length = nn.SmoothL1Loss()(pred_log_T, torch.log(t))

    total = (bce_w * bce) + (smooth_w * smooth) + (length_w * length)
    parts = {"bce": bce.detach(), "smooth": smooth.detach(), "length": length.detach()}
    return total, parts

