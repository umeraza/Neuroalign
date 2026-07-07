import torch

from neuroalign_emeg.losses import mtm_loss, tokenizer_loss
from neuroalign_emeg.metrics import reconstruction_metrics, retrieval_metrics


def test_tokenizer_loss_and_reconstruction_metrics():
    x = torch.randn(2, 4, 64)
    y = x + 0.01 * torch.randn_like(x)
    loss = tokenizer_loss(x, y, torch.tensor(0.1))
    assert loss["loss"].item() > 0
    metrics = reconstruction_metrics(x, y)
    assert metrics["mae"] >= 0
    assert -1 <= metrics["pcc"] <= 1


def test_mtm_and_retrieval_metrics():
    logits = torch.randn(2, 3, 4, 2, 8)
    target = torch.randint(0, 8, (2, 3, 4, 2))
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    loss = mtm_loss(logits, target, mask)
    assert loss.item() > 0
    emb = torch.eye(4)
    out = retrieval_metrics(emb, emb)
    assert out["recall@1"] == 1.0
