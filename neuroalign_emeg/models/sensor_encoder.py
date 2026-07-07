from __future__ import annotations

import torch
from torch import nn


class NeuroSensorEncoder(nn.Module):
    """Encodes 3D position, orientation, and sensor type."""

    def __init__(self, d_model: int = 128, num_sensor_types: int = 4, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.coord = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.type_emb = nn.Embedding(num_sensor_types, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, locs: torch.Tensor, sensor_types: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(locs.shape[0], locs.shape[1], self.coord[0].out_features, device=locs.device, dtype=locs.dtype)
        sensor_types = sensor_types.clamp_min(0).clamp_max(self.type_emb.num_embeddings - 1)
        return self.norm(self.coord(locs) + self.type_emb(sensor_types))
