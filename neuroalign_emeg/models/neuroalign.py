from __future__ import annotations

import torch
from torch import nn

from .heads import ClassificationHead, ProjectionHead
from .text_encoder import HashingClinicalTextEncoder
from .tokenizer import NeuroTokenizer
from .transformer import NeuroAlignEncoder


class NeuroAlignEMEG(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg.get("model", cfg)
        self.tokenizer = NeuroTokenizer(
            d_model=m.get("d_model", 128),
            source_tokens=m.get("source_tokens", 16),
            temporal_stride=m.get("temporal_stride", 8),
            sensor_types=m.get("sensor_types", 4),
            rvq_codebooks=m.get("rvq_codebooks", 4),
            rvq_codes=m.get("rvq_codes", 128),
            heads=m.get("transformer_heads", 4),
            dropout=m.get("dropout", 0.1),
            use_sensor_encoder=m.get("use_sensor_encoder", True),
            use_cross_attention=m.get("use_cross_attention", True),
            use_rvq=m.get("use_rvq", True),
        )
        self.encoder = NeuroAlignEncoder(
            d_model=m.get("d_model", 128),
            codebook_size=m.get("rvq_codes", 128),
            num_codebooks=m.get("rvq_codebooks", 4),
            source_tokens=m.get("source_tokens", 16),
            layers=m.get("transformer_layers", 2),
            heads=m.get("transformer_heads", 4),
            dropout=m.get("dropout", 0.1),
            separated_attention=m.get("separated_attention", True),
            use_rope=m.get("use_rope", True),
        )
        self.text_encoder = HashingClinicalTextEncoder(text_dim=m.get("text_dim", 384))
        self.signal_proj = ProjectionHead(m.get("d_model", 128), m.get("projection_dim", 128), dropout=m.get("dropout", 0.1))
        self.text_proj = ProjectionHead(m.get("text_dim", 384), m.get("projection_dim", 128), dropout=m.get("dropout", 0.1))
        self.classifier = ClassificationHead(m.get("d_model", 128), m.get("num_classes", 2), dropout=m.get("dropout", 0.1))

    def tokenize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.tokenizer(batch["signal"], batch["sensor_locs"], batch["sensor_types"], batch.get("sensor_mask"))

    def encode_from_indices(self, indices: torch.Tensor, mask_ratio: float = 0.0) -> dict[str, torch.Tensor]:
        mask = None
        if mask_ratio > 0:
            mask = self.encoder.make_random_mask(indices, mask_ratio)
        out = self.encoder(indices, mask=mask)
        out["mask"] = mask
        return out

    def forward(self, batch: dict[str, torch.Tensor], stage: str = "finetune", mask_ratio: float = 0.4) -> dict[str, torch.Tensor]:
        tok = self.tokenize_batch(batch)
        enc = self.encode_from_indices(tok["indices"], mask_ratio=mask_ratio if stage == "mtm" else 0.0)
        out = {**tok, **{f"enc_{k}": v for k, v in enc.items()}}
        out["logits"] = self.classifier(enc["pooled"])
        out["signal_embedding"] = self.signal_proj(enc["pooled"])
        if "report_segments" in batch:
            text, text_mask = self.text_encoder(batch["report_segments"], device=batch["signal"].device)
            out["text_embedding_segments"] = self.text_proj(text)
            out["text_mask"] = text_mask
            denom = text_mask.sum(dim=1, keepdim=True).clamp_min(1)
            out["text_embedding"] = (out["text_embedding_segments"] * text_mask.unsqueeze(-1)).sum(dim=1) / denom
            out["text_embedding"] = torch.nn.functional.normalize(out["text_embedding"], dim=-1)
        return out
