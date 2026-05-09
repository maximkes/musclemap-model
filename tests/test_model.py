from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import torch
from torch import nn

from src.head import ActivationHead, LengthPredictor
from src.model import MuscleMAPModel


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(768, 512, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)  # placeholder for LoRA wrapping

    def forward(self, text_tokens: Any, motion_tokens: Any | None = None) -> dict[str, torch.Tensor]:
        B = int(getattr(text_tokens, "shape", [2])[0]) if hasattr(text_tokens, "shape") else 2
        enc = torch.randn(B, 10, 768)
        dec = torch.randn(B, 8, 768)
        return {"encoder_hidden": enc, "decoder_hidden": dec}

    def generate(self, text_tokens: Any) -> dict[str, torch.Tensor]:
        return self.forward(text_tokens, motion_tokens=None)


def test_svd_warm_start_changes_input_proj() -> None:
    backbone = DummyBackbone()
    head = ActivationHead(max_T=64)
    model = MuscleMAPModel(backbone=backbone, activation_head=head, length_predictor=LengthPredictor())

    before = model.activation_head.input_proj.weight.detach().clone()
    logits, pred_log_T, _ = model(text_tokens=torch.zeros(2, 1, dtype=torch.int64), motion_tokens=torch.zeros(2, 1))
    after = model.activation_head.input_proj.weight.detach().clone()

    assert logits.ndim == 3
    assert pred_log_T.shape == (2, 1)
    assert model._svd_done is True
    assert not torch.allclose(before, after)


def test_only_head_and_predictor_trainable_before_lora() -> None:
    backbone = DummyBackbone()
    model = MuscleMAPModel(backbone=backbone, activation_head=ActivationHead(max_T=64), length_predictor=LengthPredictor())

    trainable = {id(p) for p in model.parameters_to_train()}
    assert trainable
    head_and_lp = {id(p) for p in model.activation_head.parameters()} | {
        id(p) for p in model.length_predictor.parameters()
    }
    assert head_and_lp <= trainable
    for p in backbone.parameters():
        assert p.requires_grad is False
        assert id(p) not in trainable


def test_apply_lora_adds_trainable_params() -> None:
    backbone = DummyBackbone()
    model = MuscleMAPModel(backbone=backbone, activation_head=ActivationHead(max_T=64), length_predictor=LengthPredictor())

    # Install a tiny fake peft module.
    peft = ModuleType("peft")

    class LoraConfig:  # noqa: D401
        """Fake LoraConfig."""

        def __init__(self, r: int, lora_alpha: int, lora_dropout: float, target_modules: list[str]) -> None:
            _ = (r, lora_alpha, lora_dropout, target_modules)

    def get_peft_model(module: nn.Module, _cfg: Any) -> nn.Module:
        module.lora_A = nn.Parameter(torch.zeros(1))  # type: ignore[attr-defined]
        module.lora_A.requires_grad = True  # type: ignore[attr-defined]
        return module

    peft.LoraConfig = LoraConfig  # type: ignore[attr-defined]
    peft.get_peft_model = get_peft_model  # type: ignore[attr-defined]
    sys.modules["peft"] = peft

    cfg = {"training": {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "lora_target_modules": ["q", "v"]}}
    model.apply_lora(cfg)

    assert any("lora" in n and p.requires_grad for n, p in model.named_parameters())


def test_teacher_forcing_path_shapes() -> None:
    backbone = DummyBackbone()
    cfg = {"model": {"length_predictor": {"min_T": 30, "max_T": 64}}}
    model = MuscleMAPModel(
        backbone=backbone,
        activation_head=ActivationHead(max_T=64),
        length_predictor=LengthPredictor(),
        config=cfg,
    )

    logits, pred_log_T, motion_out = model(text_tokens=torch.zeros(2, 1, dtype=torch.int64), motion_tokens=torch.zeros(2, 1))
    assert motion_out is not None
    assert pred_log_T.shape == (2, 1)
    assert logits.shape[0] == 2
    assert logits.shape[2] == 80

