from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, logits) -> dict[str, float]:
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    pred = logits.argmax(axis=-1)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }
    try:
        if logits.shape[-1] == 2:
            prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            out["auroc"] = float(roc_auc_score(y_true, prob))
            out["auprc"] = float(average_precision_score(y_true, prob))
        else:
            prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            out["auroc"] = float(roc_auc_score(y_true, prob, multi_class="ovr"))
    except Exception:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def reconstruction_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    err = y - x
    mae = err.abs().mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    x0 = x - x.mean(dim=-1, keepdim=True)
    y0 = y - y.mean(dim=-1, keepdim=True)
    pcc = ((x0 * y0).sum(dim=-1) / (x0.norm(dim=-1) * y0.norm(dim=-1)).clamp_min(1e-6)).mean().item()
    return {"mae": mae, "rmse": rmse, "pcc": pcc}


def retrieval_metrics(signal_emb: torch.Tensor, text_emb: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    sim = signal_emb @ text_emb.t()
    ranks = []
    for i in range(sim.shape[0]):
        order = torch.argsort(sim[i], descending=True)
        rank = (order == i).nonzero(as_tuple=False)[0, 0].item() + 1
        ranks.append(rank)
    ranks = np.asarray(ranks)
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["median_rank"] = float(np.median(ranks))
    out["mrr"] = float((1.0 / ranks).mean())
    return out


def seizure_event_metrics(y_true, y_pred, hours: float = 1.0) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    sens = tp / max(1.0, tp + fn)
    fa_h = fp / max(1e-6, hours)
    return {"sensitivity": sens, "false_alarms_per_hour": fa_h}
