"""Modèle A network — CNN1D backbone + BiLSTM + HRV/morpho aux branch.

Architecture
------------
    input signal (B, 12, 1000)
        │
    ┌───┴────────────┐
    │ CNN1D backbone │  3 × (Conv1d → BN → ReLU → MaxPool/2)
    │                │  channels 12 → 64 → 128 → 256
    │                │  output shape (B, 256, T')  with T' ≈ 125
    └───┬────────────┘
        │
        ▼
    BiLSTM(hidden=128)  →  last hidden state  →  (B, 256)
        │
        │       aux features (B, 12) ──► FC(32) ──┐
        ▼                                          │
        └────────────► concat ◄────────────────────┘
                        (B, 288)
                        FC(128) → Dropout → FC(3)

The final layer returns raw logits; loss / softmax are applied outside.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MODEL


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, pool: int) -> None:
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        kernels: tuple[int, ...],
        pool: int,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernels)
        blocks = []
        prev = in_channels
        for c, k in zip(channels, kernels):
            blocks.append(ConvBlock(prev, c, k, pool))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ModelA(nn.Module):
    """Final classifier returning logits of shape ``(B, n_classes)``."""

    def __init__(self, cfg=MODEL) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = CNNBackbone(
            in_channels=cfg.in_channels,
            channels=cfg.cnn_channels,
            kernels=cfg.cnn_kernels,
            pool=cfg.cnn_pool,
        )
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_channels,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.lstm_bidirectional,
        )
        lstm_out = cfg.lstm_hidden * (2 if cfg.lstm_bidirectional else 1)

        self.aux = nn.Sequential(
            nn.Linear(cfg.aux_in, cfg.aux_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.aux_hidden),
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_out + cfg.aux_hidden, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, cfg.n_classes),
        )

    def forward(
        self, signal: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        cnn_out = self.cnn(signal)
        cnn_seq = cnn_out.transpose(1, 2)
        _, (h_n, _) = self.lstm(cnn_seq)
        if self.lstm.bidirectional:
            lstm_feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            lstm_feat = h_n[-1]
        aux = self.aux(features)
        merged = torch.cat([lstm_feat, aux], dim=-1)
        return self.head(merged)

    def predict_proba(
        self, signal: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self(signal, features), dim=-1)

    def risk_score(self, probs: torch.Tensor) -> torch.Tensor:
        """Continuous risk score in [0, 1]: P(critique) + 0.5 * P(suspect)."""
        return probs[..., 2] + 0.5 * probs[..., 1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _self_test() -> None:
    model = ModelA()
    n_params = count_parameters(model)
    sig = torch.randn(4, 12, 1000)
    feats = torch.randn(4, 12)
    logits = model(sig, feats)
    probs = model.predict_proba(sig, feats)
    risks = model.risk_score(probs)
    print(f"params={n_params:,}")
    print(f"logits.shape={tuple(logits.shape)}  probs.sum={probs.sum(dim=-1)}")
    print(f"risk_scores={risks.tolist()}")
    assert logits.shape == (4, 3)
    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)
    print("model self-test OK")


if __name__ == "__main__":
    _self_test()
