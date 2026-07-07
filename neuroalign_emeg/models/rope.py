from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply lightweight rotary temporal encoding to [..., W, D]."""
    d = x.shape[-1]
    if d % 2:
        return x
    w = x.shape[-2]
    device = x.device
    pos = torch.arange(w, device=device, dtype=x.dtype)
    freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device, dtype=x.dtype) / d))
    angles = pos[:, None] * freq[None, :]
    cos = angles.cos()
    sin = angles.sin()
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
