# Paper result targets

Use this file as a checklist when reproducing the paper tables.

## Main performance comparison

| Model | TUAB Acc | TUAB AUROC | TUEV Macro-F1 | TUSZ Sens/FA-h | SomatoMotor Acc | Report R@1/MRR |
|---|---:|---:|---:|---:|---:|---:|
| NeuroAlign-EMEG target | 90.38 | 95.41 | 74.92 | 87.46 / 0.24 | 87.92 | 41.67 / 0.563 |

## Required ablations

- Remove NeuroSensor Encoder.
- Remove cross-attention compression.
- Remove both sensor and cross-attention modules.
- Signal-only pretraining without report alignment.
- One-to-one contrastive alignment instead of MIL.
- Remove orthogonality regularization.
- Joint full spatiotemporal attention.
- Standard positional encoding instead of RoPE.
- Continuous latent model without RVQ.
- Remove frequency and PCC reconstruction losses.
