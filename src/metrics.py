from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(pred: np.ndarray, true: np.ndarray, muscle_names: list[str]) -> dict[str, Any]:
    """Compute evaluation metrics for muscle activations.

    Args:
        pred: Predicted activations, shape [B, T, N] or [T, N]
        true: Ground truth activations, same shape as pred
        muscle_names: Names of muscles, length N
    """

    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have same shape, got {pred.shape} vs {true.shape}")
    if pred.ndim == 2:
        pred = pred[None, ...]
        true = true[None, ...]
    if pred.ndim != 3:
        raise ValueError(f"Expected pred/true with ndim 2 or 3, got {pred.ndim}")
    B, T, N = pred.shape
    if len(muscle_names) != N:
        raise ValueError(f"Expected {N} muscle names, got {len(muscle_names)}")

    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias_mean = float(np.mean(diff))

    # Smoothness: mean absolute temporal derivative
    if T >= 2:
        smooth_pred = float(np.mean(np.abs(pred[:, 1:] - pred[:, :-1])))
        smooth_true = float(np.mean(np.abs(true[:, 1:] - true[:, :-1])))
    else:
        smooth_pred = 0.0
        smooth_true = 0.0

    # Pearson r per muscle, averaged.
    x = pred.reshape(-1, N)
    y = true.reshape(-1, N)
    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = y - y.mean(axis=0, keepdims=True)
    num = np.sum(x_center * y_center, axis=0)
    den = np.sqrt(np.sum(x_center**2, axis=0) * np.sum(y_center**2, axis=0)) + 1e-8
    pearson_r = num / den
    pearson_r_mean = float(np.mean(pearson_r))

    # R2 per muscle.
    ss_res = np.sum((x - y) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0) + 1e-8
    r2 = 1.0 - (ss_res / ss_tot)

    order = np.argsort(r2)
    bottom_idx = order[:10].tolist()
    top_idx = order[::-1][:10].tolist()

    top10 = [{"muscle": muscle_names[i], "r2": float(r2[i])} for i in top_idx]
    bottom10 = [{"muscle": muscle_names[i], "r2": float(r2[i])} for i in bottom_idx]

    return {
        "mpjae": mae,  # legacy name in spec; here = mean per-element abs error on activations
        "mae": mae,
        "rmse": rmse,
        "mean_bias": bias_mean,
        "pearson_r_mean": pearson_r_mean,
        "pearson_r_per_muscle": pearson_r.astype(np.float32),
        "r2_mean": float(np.mean(r2)),
        "r2_per_muscle": r2.astype(np.float32),
        "r2_top10": top10,
        "r2_bottom10": bottom10,
        "smoothness_pred": smooth_pred,
        "smoothness_true": smooth_true,
        "length_mae": 0.0,
        "shapes": {"B": int(B), "T": int(T), "N": int(N)},
    }

