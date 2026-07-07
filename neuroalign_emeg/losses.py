from __future__ import annotations

import torch
import torch.nn.functional as F


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    return (x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(eps)


def spectral_losses(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = torch.fft.rfft(x.float(), dim=-1)
    yf = torch.fft.rfft(y.float(), dim=-1)
    amp = F.l1_loss(xf.abs(), yf.abs())
    phase = F.l1_loss(torch.angle(xf), torch.angle(yf))
    return amp, phase


def tokenizer_loss(signal: torch.Tensor, recon: torch.Tensor, commitment_loss: torch.Tensor, lambda_freq: float = 0.25, lambda_pcc: float = 0.10, lambda_rvq: float = 1.0) -> dict[str, torch.Tensor]:
    time = F.l1_loss(recon, signal)
    amp, phase = spectral_losses(signal, recon)
    pcc_val = pearson_corr(signal, recon).mean()
    pcc_loss = torch.exp(-pcc_val)
    total = time + lambda_freq * (amp + phase) + lambda_pcc * pcc_loss + lambda_rvq * commitment_loss
    return {"loss": total, "time": time, "freq_amp": amp, "freq_phase": phase, "pcc_loss": pcc_loss, "rvq": commitment_loss}


def mtm_loss(logits: torch.Tensor, target_indices: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    # logits [B,C,W,Nq,K], target [B,C,W,Nq]
    if mask is None:
        mask = torch.ones(target_indices.shape[:-1], device=target_indices.device, dtype=torch.bool)
    if mask.sum() == 0:
        return logits.sum() * 0
    losses = []
    for j in range(target_indices.shape[-1]):
        pred = logits[..., j, :][mask]
        tgt = target_indices[..., j][mask]
        losses.append(F.cross_entropy(pred, tgt))
    return torch.stack(losses).mean()


def mil_alignment_loss(signal_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # Batch-level bidirectional MIL simplified to same-record positives.
    s = signal_emb @ text_emb.t() / temperature
    labels = torch.arange(s.shape[0], device=s.device)
    return 0.5 * (F.cross_entropy(s, labels) + F.cross_entropy(s.t(), labels))


def orthogonality_loss(emb: torch.Tensor) -> torch.Tensor:
    emb = F.normalize(emb, dim=-1)
    c = emb.t() @ emb / max(1, emb.shape[0])
    eye = torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
    return ((c - eye) ** 2).sum()
