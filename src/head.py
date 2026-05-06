from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

DEFAULT_EMBED_DIM: Final[int] = 256


class LengthPredictor(nn.Module):
    """Predict log sequence length (T_frame) from encoder hidden states (stub)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(768, 128)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(128, 1)

        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, encoder_hidden: Tensor) -> Tensor:
        """Return log-length prediction as shape [B, 1]."""

        if encoder_hidden.ndim != 3 or encoder_hidden.shape[-1] != 768:
            raise ValueError(f"encoder_hidden must be [B, T_enc, 768], got {tuple(encoder_hidden.shape)}")

        pooled = encoder_hidden.mean(dim=1)  # [B, 768]
        x = self.fc1(pooled)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)  # [B, 1]

    def predict_T(self, encoder_hidden: Tensor, min_T: int, max_T: int) -> Tensor:
        """Return predicted frame length as int tensor [B]."""

        if min_T <= 0 or max_T <= 0 or min_T > max_T:
            raise ValueError(f"Invalid min_T/max_T: {min_T}/{max_T}")

        pred_log_T = self.forward(encoder_hidden)  # [B, 1]
        pred_T = torch.exp(pred_log_T).round().clamp(min=float(min_T), max=float(max_T))
        return pred_T.to(dtype=torch.int64).squeeze(-1)  # [B]


class ActivationHead(nn.Module):
    """Map decoder hidden states to per-frame muscle logits."""

    def __init__(
        self,
        *,
        n_muscles: int = 80,
        t5_hidden_dim: int = 768,
        proj_dim: int = 256,
        n_heads: int = 4,
        n_transformer_layers: int = 3,
        dropout: float = 0.1,
        max_T: int = 512,
    ) -> None:
        super().__init__()
        if proj_dim % n_heads != 0:
            raise ValueError(f"proj_dim must be divisible by n_heads: {proj_dim}/{n_heads}")
        if max_T <= 0:
            raise ValueError("max_T must be positive")

        self.n_muscles = int(n_muscles)
        self.t5_hidden_dim = int(t5_hidden_dim)
        self.proj_dim = int(proj_dim)
        self.max_T = int(max_T)

        self.input_proj = nn.Linear(self.t5_hidden_dim, self.proj_dim)
        self.query_pos = nn.Parameter(torch.empty(self.max_T, self.proj_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.proj_dim, num_heads=n_heads, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.proj_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.output_proj = nn.Linear(self.proj_dim, self.n_muscles, bias=False)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.query_pos, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, decoder_hidden: Tensor, T_frame: int) -> Tensor:
        """Return logits as shape [B, T_frame, N_muscles]."""

        if decoder_hidden.ndim != 3 or decoder_hidden.shape[-1] != self.t5_hidden_dim:
            raise ValueError(
                f"decoder_hidden must be [B, T_tok, {self.t5_hidden_dim}], got {tuple(decoder_hidden.shape)}"
            )
        if T_frame <= 0:
            raise ValueError("T_frame must be positive")
        if T_frame > self.max_T:
            raise ValueError(f"T_frame={T_frame} exceeds max_T={self.max_T}")

        B = int(decoder_hidden.shape[0])
        kv = self.input_proj(decoder_hidden)  # [B, T_tok, proj_dim]
        q = self.query_pos[:T_frame].unsqueeze(0).expand(B, -1, -1)  # [B, T_frame, proj_dim]

        x, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)  # [B, T_frame, proj_dim]
        x = self.encoder(x)  # [B, T_frame, proj_dim]
        return self.output_proj(x)  # [B, T_frame, N_muscles]

