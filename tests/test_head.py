from __future__ import annotations

import torch

from src.head import ActivationHead, LengthPredictor


class TestLengthPredictor:
    def test_forward_shape(self) -> None:
        m = LengthPredictor()
        enc = torch.randn(2, 10, 768)
        out = m(enc)
        assert out.shape == (2, 1)

    def test_predict_T_clamped(self) -> None:
        m = LengthPredictor()
        enc = torch.randn(2, 10, 768)
        pred_T = m.predict_T(enc, min_T=30, max_T=256)
        assert pred_T.shape == (2,)
        assert pred_T.dtype == torch.int64
        assert int(pred_T.min()) >= 30
        assert int(pred_T.max()) <= 256

    def test_gradients_flow(self) -> None:
        m = LengthPredictor()
        enc = torch.randn(2, 10, 768, requires_grad=True)
        out = m(enc)
        loss = out.sum()
        loss.backward()
        assert m.fc1.weight.grad is not None
        assert m.fc2.weight.grad is not None
        assert enc.grad is not None


class TestActivationHead:
    def test_forward_shapes(self) -> None:
        m = ActivationHead(max_T=64)
        hidden = torch.randn(2, 8, 768)
        out16 = m(hidden, T_frame=16)
        out32 = m(hidden, T_frame=32)
        assert out16.shape == (2, 16, 80)
        assert out32.shape == (2, 32, 80)

    def test_no_sigmoid_outputs_can_be_negative(self) -> None:
        m = ActivationHead(max_T=64)
        hidden = torch.randn(2, 8, 768)
        out = m(hidden, T_frame=16)
        assert float(out.min()) < 0.0

    def test_gradients_flow(self) -> None:
        m = ActivationHead(max_T=64)
        hidden = torch.randn(2, 8, 768, requires_grad=True)
        out = m(hidden, T_frame=16)
        loss = out.mean()
        loss.backward()
        assert m.input_proj.weight.grad is not None
        assert m.output_proj.weight.grad is not None
        assert m.query_pos.grad is not None
        assert hidden.grad is not None

