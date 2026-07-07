from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    # Limit CPU thread oversubscription; tiny EEG batches can become painfully
    # slow on shared runners when PyTorch grabs every available core.
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    apply_flat_overrides(cfg)
    return cfg


def apply_flat_overrides(cfg: dict[str, Any]) -> None:
    flat = cfg.pop("overrides", None)
    if not flat:
        return
    for dotted, value in flat.items():
        parts = dotted.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, **extra: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, strict: bool = False) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected), "payload": payload}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
