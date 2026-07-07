from __future__ import annotations

import torch
from torch import nn

from .cross_attention import CrossAttentionCompressor, SensorConditionedDecoder
from .rvq import ResidualVectorQuantizer
from .sensor_encoder import NeuroSensorEncoder
from .temporal_encoder import TemporalEncoder


class NeuroTokenizer(nn.Module):
    """Sensor-aware tokenizer with RVQ and sensor-conditioned reconstruction."""

    def __init__(
        self,
        d_model: int = 128,
        source_tokens: int = 16,
        temporal_stride: int = 8,
        sensor_types: int = 4,
        rvq_codebooks: int = 4,
        rvq_codes: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        use_sensor_encoder: bool = True,
        use_cross_attention: bool = True,
        use_rvq: bool = True,
    ) -> None:
        super().__init__()
        self.use_rvq = use_rvq
        self.temporal = TemporalEncoder(d_model=d_model, stride=temporal_stride, dropout=dropout)
        self.sensor = NeuroSensorEncoder(d_model=d_model, num_sensor_types=sensor_types, enabled=use_sensor_encoder)
        self.compressor = CrossAttentionCompressor(source_tokens, d_model, heads=heads, enabled=use_cross_attention)
        self.rvq = ResidualVectorQuantizer(d_model, rvq_codebooks, rvq_codes)
        self.decoder = SensorConditionedDecoder(d_model, heads=heads, temporal_stride=temporal_stride)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        signal: torch.Tensor,
        sensor_locs: torch.Tensor,
        sensor_types: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_time = self.temporal(signal)
        sensor_emb = self.sensor(sensor_locs, sensor_types)
        z_src = self.norm(self.compressor(z_time, sensor_emb, sensor_mask=sensor_mask))
        if self.use_rvq:
            q = self.rvq(z_src)
            z_quant = q["quantized"]
            indices = q["indices"]
            commitment = q["commitment_loss"]
        else:
            z_quant = z_src
            indices = torch.zeros(*z_src.shape[:-1], self.rvq.num_codebooks, device=z_src.device, dtype=torch.long)
            commitment = torch.zeros((), device=z_src.device, dtype=z_src.dtype)
        recon = self.decoder(z_quant, sensor_emb, target_time=signal.shape[-1])
        return {
            "z_time": z_time,
            "z_src": z_src,
            "z_quant": z_quant,
            "indices": indices,
            "reconstruction": recon,
            "commitment_loss": commitment,
        }
