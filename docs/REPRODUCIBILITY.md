# Reproducibility notes

## What is implemented

- NeuroSensor Encoder using coordinates, orientations, and sensor-type embeddings.
- Cross-attention compression from variable channels to fixed neuro-source tokens.
- Residual vector quantization with codebook statistics.
- Masked neuro-token modeling loss over RVQ code indices.
- Bidirectional neuro-text contrastive/MIL-style alignment.
- Orthogonality regularization for collapse resistance.
- Classification, reconstruction, retrieval, and event metrics.
- Ablation configs matching the paper sections.

## What is not bundled

- TUH EEG corpus files.
- OpenNeuro EMEG files.
- Private clinical reports.
- Trained checkpoints.
- Exact released tables from hidden experimental runs.

## Minimum command sequence

```bash
python scripts/make_synthetic_manifest.py
python scripts/train_tokenizer.py --config configs/base.yaml
python scripts/pretrain_mtm.py --config configs/base.yaml
python scripts/train_report_alignment.py --config configs/base.yaml
python scripts/finetune.py --config configs/base.yaml
```
