from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=-1)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
