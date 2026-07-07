from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn


class HashingClinicalTextEncoder(nn.Module):
    """Frozen local text encoder for report segments.

    This avoids downloading clinical language models during smoke tests. Replace
    with a HuggingFace encoder in real experiments if permitted by your data use
    agreement.
    """

    def __init__(self, text_dim: int = 384, vocab_buckets: int = 8192) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.vocab_buckets = vocab_buckets
        self.proj = nn.Linear(vocab_buckets, text_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        for p in self.parameters():
            p.requires_grad = False

    def _encode_one(self, text: str, device: torch.device) -> torch.Tensor:
        vec = torch.zeros(self.vocab_buckets, device=device)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.vocab_buckets
            vec[h] += 1.0
        if vec.sum() > 0:
            vec = vec / vec.norm().clamp_min(1e-6)
        return self.proj(vec)

    def forward(self, reports: list[list[str]], device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        max_m = max([len(x) for x in reports] + [1])
        out = torch.zeros(len(reports), max_m, self.text_dim, device=device)
        mask = torch.zeros(len(reports), max_m, dtype=torch.bool, device=device)
        for i, segs in enumerate(reports):
            for j, seg in enumerate(segs[:max_m]):
                out[i, j] = self._encode_one(seg, device)
                mask[i, j] = True
        return out, mask
