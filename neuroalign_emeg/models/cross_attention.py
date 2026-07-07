from __future__ import annotations

import torch
from torch import nn


class CrossAttentionCompressor(nn.Module):
    """Variable-channel to fixed neuro-source compression."""

    def __init__(self, source_tokens: int, d_model: int, heads: int = 4, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.source_tokens = source_tokens
        self.query = nn.Parameter(torch.randn(source_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fallback = nn.Linear(d_model, d_model)

    def forward(self, z_time: torch.Tensor, sensor_emb: torch.Tensor, sensor_mask: torch.Tensor | None = None) -> torch.Tensor:
        # z_time: [B,C,W,D], sensor_emb: [B,C,D]
        b, c, w, d = z_time.shape
        if not self.enabled:
            pooled = z_time.mean(dim=1)  # [B,W,D]
            pooled = self.fallback(pooled)
            return pooled.unsqueeze(1).repeat(1, self.source_tokens, 1, 1)
        keys = z_time + sensor_emb.unsqueeze(2)
        keys = keys.permute(0, 2, 1, 3).reshape(b * w, c, d)
        values = z_time.permute(0, 2, 1, 3).reshape(b * w, c, d)
        q = self.query.unsqueeze(0).expand(b * w, -1, -1)
        mask = None
        if sensor_mask is not None:
            mask = ~sensor_mask.unsqueeze(1).expand(b, w, c).reshape(b * w, c)
        out, _ = self.attn(q, keys, values, key_padding_mask=mask, need_weights=False)
        out = self.norm(out)
        return out.reshape(b, w, self.source_tokens, d).permute(0, 2, 1, 3)


class SensorConditionedDecoder(nn.Module):
    """Fixed neuro-source features back to channel-time waveform."""

    def __init__(self, d_model: int, heads: int = 4, temporal_stride: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.to_wave = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, temporal_stride))
        self.temporal_stride = temporal_stride

    def forward(self, src: torch.Tensor, sensor_emb: torch.Tensor, target_time: int) -> torch.Tensor:
        # src [B,Csrc,W,D], sensor_emb [B,C,D]
        b, csrc, w, d = src.shape
        c = sensor_emb.shape[1]
        memory = src.permute(0, 2, 1, 3).reshape(b * w, csrc, d)
        queries = sensor_emb.unsqueeze(1).expand(b, w, c, d).reshape(b * w, c, d)
        out, _ = self.attn(queries, memory, memory, need_weights=False)
        out = self.norm(out).reshape(b, w, c, d).permute(0, 2, 1, 3)
        wave = self.to_wave(out).reshape(b, c, w * self.temporal_stride)
        if wave.shape[-1] < target_time:
            wave = torch.nn.functional.pad(wave, (0, target_time - wave.shape[-1]))
        return wave[..., :target_time]
