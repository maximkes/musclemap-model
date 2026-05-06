from __future__ import annotations

from typing import Any

import torch

from src.losses import activation_loss


def _base_config() -> dict[str, Any]:
    return {"training": {"loss": {"bce_weight": 1.0, "smoothness_weight": 0.1, "length_weight": 0.1}}}


def test_loss_near_zero_when_probs_match_targets_and_constant_over_time() -> None:
    B, T, N = 2, 16, 80
    # For BCEWithLogits, near-zero loss occurs when targets are {0,1}
    # and logits have the correct large sign.
    targets = torch.zeros((B, T, N), dtype=torch.float32)
    logits = torch.full((B, T, N), -20.0, dtype=torch.float32)
    mask = torch.ones((B, T), dtype=torch.bool)

    total, parts = activation_loss(logits, targets, mask, _base_config())
    assert float(total) < 1e-5
    assert float(parts["bce"]) < 1e-5
    assert float(parts["smooth"]) < 1e-5


def test_mask_zeros_out_padded_frames_in_bce() -> None:
    B, T, N = 2, 16, 80
    targets = torch.zeros((B, T, N), dtype=torch.float32)
    logits = torch.zeros((B, T, N), dtype=torch.float32)

    # Make padded region very wrong; it should not affect loss.
    logits[:, 8:] = 100.0
    mask = torch.zeros((B, T), dtype=torch.bool)
    mask[:, :8] = True

    total_masked, _ = activation_loss(logits, targets, mask, _base_config())

    logits_trunc = logits[:, :8]
    targets_trunc = targets[:, :8]
    mask_trunc = torch.ones((B, 8), dtype=torch.bool)
    total_trunc, _ = activation_loss(logits_trunc, targets_trunc, mask_trunc, _base_config())

    assert torch.allclose(total_masked, total_trunc, atol=1e-6, rtol=0.0)

