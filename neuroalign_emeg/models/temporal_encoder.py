from __future__ import annotations

import torch
from torch import nn


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.norm(x + self.net(x)))


class TemporalEncoder(nn.Module):
    """Per-sensor temporal encoder producing [B, C, W, D]."""

    def __init__(self, d_model: int = 128, stride: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.stride = stride
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, stride=stride, padding=3),
            nn.GELU(),
            ConvResidualBlock(d_model, dropout=dropout),
            ConvResidualBlock(d_model, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = self.stem(x.reshape(b * c, 1, t))
        y = y.transpose(1, 2).reshape(b, c, y.shape[-1], -1)
        return y
