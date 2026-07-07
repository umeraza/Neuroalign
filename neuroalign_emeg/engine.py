from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import mil_alignment_loss, mtm_loss, orthogonality_loss, tokenizer_loss
from .metrics import classification_metrics, reconstruction_metrics, retrieval_metrics
from .utils import save_checkpoint


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    o = cfg.get("optim", cfg)
    return torch.optim.AdamW(model.parameters(), lr=o.get("lr", 3e-4), weight_decay=o.get("weight_decay", 0.05))


def _mean_logs(logs: list[dict[str, float]]) -> dict[str, float]:
    acc = defaultdict(list)
    for row in logs:
        for k, v in row.items():
            try:
                acc[k].append(float(v))
            except Exception:
                pass
    return {k: sum(v) / max(1, len(v)) for k, v in acc.items()}


def train_tokenizer_epoch(model, loader: DataLoader, optimizer, device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    loss_cfg = cfg.get("loss", {})
    for batch in tqdm(loader, desc="tokenizer", leave=False):
        batch = move_batch(batch, device)
        out = model(batch, stage="tokenizer")
        losses = tokenizer_loss(batch["signal"], out["reconstruction"], out["commitment_loss"], loss_cfg.get("lambda_freq", 0.25), loss_cfg.get("lambda_pcc", 0.1), loss_cfg.get("lambda_rvq", 1.0))
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("optim", {}).get("grad_clip", 1.0))
        optimizer.step()
        logs.append({k: v.detach().item() for k, v in losses.items()})
    return _mean_logs(logs)


def train_mtm_epoch(model, loader: DataLoader, optimizer, device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    mask_ratio = cfg.get("loss", {}).get("mask_ratio", 0.4)
    for batch in tqdm(loader, desc="mtm", leave=False):
        batch = move_batch(batch, device)
        out = model(batch, stage="mtm", mask_ratio=mask_ratio)
        loss = mtm_loss(out["enc_logits"], out["indices"], out["enc_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("optim", {}).get("grad_clip", 1.0))
        optimizer.step()
        logs.append({"loss": loss.detach().item()})
    return _mean_logs(logs)


def train_alignment_epoch(model, loader: DataLoader, optimizer, device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    loss_cfg = cfg.get("loss", {})
    for batch in tqdm(loader, desc="alignment", leave=False):
        batch = move_batch(batch, device)
        out = model(batch, stage="align")
        mil = mil_alignment_loss(out["signal_embedding"], out["text_embedding"], loss_cfg.get("temperature", 0.07))
        orth = orthogonality_loss(out["signal_embedding"])
        loss = loss_cfg.get("beta_mil", 1.0) * mil + loss_cfg.get("gamma_orth", 0.05) * orth
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logs.append({"loss": loss.detach().item(), "mil": mil.detach().item(), "orth": orth.detach().item()})
    return _mean_logs(logs)


def train_finetune_epoch(model, loader: DataLoader, optimizer, device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    for batch in tqdm(loader, desc="finetune", leave=False):
        batch = move_batch(batch, device)
        out = model(batch, stage="finetune")
        loss = torch.nn.functional.cross_entropy(out["logits"], batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logs.append({"loss": loss.detach().item()})
    return _mean_logs(logs)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device) -> dict[str, float]:
    model.eval()
    ys, logits = [], []
    sigs, txts = [], []
    recon_logs = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        out = model(batch, stage="eval")
        ys.extend(batch["label"].cpu().tolist())
        logits.append(out["logits"].detach().cpu())
        if "text_embedding" in out:
            sigs.append(out["signal_embedding"].detach().cpu())
            txts.append(out["text_embedding"].detach().cpu())
        recon_logs.append(reconstruction_metrics(batch["signal"].detach().cpu(), out["reconstruction"].detach().cpu()))
    metrics = classification_metrics(ys, torch.cat(logits, dim=0).numpy())
    if sigs and txts:
        metrics.update({"retrieval_" + k: v for k, v in retrieval_metrics(torch.cat(sigs), torch.cat(txts)).items()})
    metrics.update({"recon_" + k: v for k, v in _mean_logs(recon_logs).items()})
    return metrics


def move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def fit_loop(model, train_loader, val_loader, optimizer, device, cfg: dict, train_fn: Callable, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(cfg.get("optim", {}).get("epochs", 1))
    for epoch in range(epochs):
        train_logs = train_fn(model, train_loader, optimizer, device, cfg)
        val_logs = evaluate(model, val_loader, device) if val_loader is not None else {}
        print({"epoch": epoch + 1, "train": train_logs, "val": val_logs})
        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch=epoch + 1, train=train_logs, val=val_logs)
