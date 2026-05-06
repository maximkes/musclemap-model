from __future__ import annotations

import numpy as np

from src.metrics import compute_metrics


def test_metrics_shapes_and_keys() -> None:
    B, T, N = 4, 16, 80
    pred = np.zeros((B, T, N), dtype=np.float32)
    true = np.zeros((B, T, N), dtype=np.float32)
    names = [f"m{i}" for i in range(N)]

    out = compute_metrics(pred, true, names)
    assert out["shapes"] == {"B": B, "T": T, "N": N}
    assert out["r2_per_muscle"].shape == (N,)
    assert out["pearson_r_per_muscle"].shape == (N,)
    assert len(out["r2_top10"]) == 10
    assert len(out["r2_bottom10"]) == 10

    for k in ("mae", "rmse", "mean_bias", "pearson_r_mean", "r2_mean", "smoothness_pred", "smoothness_true"):
        assert k in out


def test_r2_is_one_when_pred_equals_true() -> None:
    B, T, N = 4, 16, 80
    true = np.random.RandomState(0).rand(B, T, N).astype(np.float32)
    pred = true.copy()
    names = [f"m{i}" for i in range(N)]

    out = compute_metrics(pred, true, names)
    assert np.allclose(out["r2_per_muscle"], 1.0, atol=1e-5, rtol=0.0)

