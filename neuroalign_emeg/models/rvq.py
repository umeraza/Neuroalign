from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """Straight-through residual vector quantization."""

    def __init__(self, d_model: int = 128, num_codebooks: int = 4, codebook_size: int = 128) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, d_model) * 0.02)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # z [B,C,W,D]
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        commit = torch.zeros((), device=z.device, dtype=z.dtype)
        for j in range(self.num_codebooks):
            codebook = self.codebooks[j]
            flat = residual.reshape(-1, residual.shape[-1])
            dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ codebook.t() + codebook.pow(2).sum(1).unsqueeze(0))
            idx = dist.argmin(dim=1)
            q = F.embedding(idx, codebook).reshape_as(residual)
            quantized_sum = quantized_sum + q
            commit = commit + F.mse_loss(residual.detach(), q) + 0.25 * F.mse_loss(residual, q.detach())
            residual = residual - q.detach()
            all_indices.append(idx.reshape(z.shape[:-1]))
        quantized = z + (quantized_sum - z).detach()
        return {"quantized": quantized, "indices": torch.stack(all_indices, dim=-1), "commitment_loss": commit / self.num_codebooks}

    @torch.no_grad()
    def codebook_stats(self, indices: torch.Tensor) -> dict[str, float]:
        vals = indices.reshape(-1)
        hist = torch.bincount(vals, minlength=self.codebook_size).float()
        probs = hist / hist.sum().clamp_min(1)
        perplexity = torch.exp(-(probs * (probs + 1e-12).log()).sum()).item()
        utilization = (hist > 0).float().mean().item()
        return {"codebook_perplexity": perplexity, "code_utilization": utilization}
