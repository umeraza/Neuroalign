from __future__ import annotations

import torch
from torch import nn

from .rope import apply_rope


class SeparatedSpatialTemporalBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, use_rope: bool = True) -> None:
        super().__init__()
        self.use_rope = use_rope
        self.temporal = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.spatial = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,Csrc,W,D]
        b, c, w, d = x.shape
        xt = x.reshape(b * c, w, d)
        if self.use_rope:
            xt = apply_rope(xt)
        yt, _ = self.temporal(xt, xt, xt, need_weights=False)
        x = self.norm_t((xt + yt).reshape(b, c, w, d))
        xs = x.permute(0, 2, 1, 3).reshape(b * w, c, d)
        ys, _ = self.spatial(xs, xs, xs, need_weights=False)
        x = self.norm_s((xs + ys).reshape(b, w, c, d).permute(0, 2, 1, 3))
        return x + self.ffn(x)


class JointAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, use_rope: bool = True) -> None:
        super().__init__()
        self.use_rope = use_rope
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, d = x.shape
        z = x.reshape(b, c * w, d)
        y, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm(z + y).reshape(b, c, w, d)
        return z + self.ffn(z)


class NeuroAlignEncoder(nn.Module):
    """Masked neuro-token spatiotemporal transformer."""

    def __init__(
        self,
        d_model: int = 128,
        codebook_size: int = 128,
        num_codebooks: int = 4,
        source_tokens: int = 16,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        separated_attention: bool = True,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.source_tokens = source_tokens
        self.embeddings = nn.ModuleList([nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks)])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        block = SeparatedSpatialTemporalBlock if separated_attention else JointAttentionBlock
        self.blocks = nn.ModuleList([block(d_model, heads, dropout=dropout, use_rope=use_rope) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, codebook_size) for _ in range(num_codebooks)])

    def embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z = 0
        for j, emb in enumerate(self.embeddings):
            z = z + emb(indices[..., j].clamp_min(0).clamp_max(self.codebook_size - 1))
        return z / max(1, self.num_codebooks)

    def forward(self, indices: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.embed_indices(indices)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(dtype=x.dtype), x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = torch.stack([head(x) for head in self.heads], dim=-2)
        pooled = x.mean(dim=(1, 2))
        return {"tokens": x, "pooled": pooled, "logits": logits}

    @staticmethod
    def make_random_mask(indices: torch.Tensor, mask_ratio: float = 0.4) -> torch.Tensor:
        return torch.rand(indices.shape[:-1], device=indices.device) < mask_ratio
