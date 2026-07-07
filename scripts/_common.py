from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from neuroalign_emeg.data import ManifestEMEGDataset, SyntheticEMEGDataset, emeg_collate
from neuroalign_emeg.models import NeuroAlignEMEG
from neuroalign_emeg.engine import make_optimizer
from neuroalign_emeg.utils import get_device, load_checkpoint, load_yaml, set_seed


def build_loaders(cfg):
    data = cfg.get("data", {})
    batch_size = cfg.get("optim", {}).get("batch_size", 4)
    if data.get("synthetic", True) or not data.get("train_manifest"):
        ds = SyntheticEMEGDataset(
            num_samples=data.get("num_samples", 64),
            num_classes=data.get("num_classes", cfg.get("model", {}).get("num_classes", 2)),
            max_channels=data.get("max_channels", 64),
            max_time=data.get("max_time", 512),
            num_sensor_types=data.get("num_sensor_types", 4),
            use_reports=data.get("use_reports", True),
            seed=cfg.get("seed", 42),
        )
        n_val = max(1, len(ds) // 5)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(cfg.get("seed", 42)))
    else:
        train_ds = ManifestEMEGDataset(data["train_manifest"], max_time=data.get("max_time"))
        val_ds = ManifestEMEGDataset(data["val_manifest"], max_time=data.get("max_time")) if data.get("val_manifest") else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=data.get("num_workers", 0), collate_fn=emeg_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=data.get("num_workers", 0), collate_fn=emeg_collate) if val_ds is not None else None
    return train_loader, val_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tokenizer-checkpoint", default=None)
    p.add_argument("--encoder-checkpoint", default=None)
    return p.parse_args()
