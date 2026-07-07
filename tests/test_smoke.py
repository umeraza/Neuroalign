from torch.utils.data import DataLoader

from neuroalign_emeg.data import SyntheticEMEGDataset, emeg_collate
from neuroalign_emeg.models import NeuroAlignEMEG


def tiny_cfg():
    return {
        "model": {
            "d_model": 32,
            "source_tokens": 4,
            "temporal_stride": 8,
            "sensor_types": 4,
            "rvq_codebooks": 2,
            "rvq_codes": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "num_classes": 2,
            "text_dim": 64,
            "projection_dim": 32,
        }
    }


def test_forward_shapes():
    ds = SyntheticEMEGDataset(num_samples=4, max_channels=12, max_time=128, num_classes=2)
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=emeg_collate)))
    model = NeuroAlignEMEG(tiny_cfg())
    # adjust text encoder dim in this tiny config is handled by constructor
    out = model(batch, stage="mtm", mask_ratio=0.5)
    assert out["reconstruction"].shape == batch["signal"].shape
    assert out["indices"].shape[-1] == 2
    assert out["enc_logits"].shape[-2] == 2
    assert out["logits"].shape == (2, 2)
