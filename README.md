# NeuroAlign-EMEG

Reference PyTorch implementation scaffold for **NeuroAlign-EMEG: Sensor-Aware EEG-MEG Foundation Learning with Clinical Report Alignment for Neurodiagnostic Modeling**.

This repository is structured around the paper's four-stage training strategy:

1. **NeuroTokenizer training** with time, frequency, PCC, and RVQ commitment losses.
2. **Masked neuro-token modeling** over the fixed neuro-source-time grid.
3. **Clinical report-guided neuro-text alignment** with bidirectional multi-instance learning and orthogonality regularization.
4. **Downstream adaptation** for TUAB, TUEV, TUSZ, SomatoMotor, and report retrieval.

> Important: this repo is a reproducibility-ready scaffold. It does not contain TUH/OpenNeuro datasets, private EEG reports, trained checkpoints, or the exact hidden preprocessing manifests used to produce the paper tables. It includes dataset adapters, config files, scripts, metrics, ablation switches, and synthetic smoke tests so the code runs immediately.

## Repository layout

```text
neuroalign-emeg/
├── configs/                  # YAML configs for all stages and ablations
├── docs/                     # dataset, model-card, results, reproducibility notes
├── examples/                 # smoke-run shell script
├── neuroalign_emeg/          # package source
│   ├── data/                 # manifest datasets, preprocessing, reports, collate
│   ├── models/               # tokenizer, RVQ, sensor encoder, transformer, heads
│   ├── engine.py             # reusable training/evaluation loops
│   ├── losses.py             # L_tok, L_mtm, L_mil, L_orth, clinical loss helpers
│   ├── metrics.py            # classification, seizure, reconstruction, retrieval metrics
│   └── utils.py              # config, seed, checkpoint helpers
├── scripts/                  # train/evaluate/ablation entry points
├── tests/                    # synthetic smoke tests
├── tools/                    # dataset manifest builders and checkpoint export helpers
├── pyproject.toml
└── requirements.txt
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Smoke test without data

```bash
pytest -q
bash examples/smoke_run.sh
```

## Stage 1: train NeuroTokenizer

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
```

## Stage 2: masked neuro-token pretraining

```bash
python scripts/pretrain_mtm.py --config configs/mtm_pretrain.yaml \
  --tokenizer-checkpoint outputs/tokenizer/checkpoint_last.pt
```

## Stage 3: report-guided alignment

```bash
python scripts/train_report_alignment.py --config configs/report_align.yaml \
  --encoder-checkpoint outputs/mtm/checkpoint_last.pt
```

## Stage 4: downstream fine-tuning

```bash
python scripts/finetune.py --config configs/finetune_tuab.yaml
python scripts/finetune.py --config configs/finetune_tuev.yaml
python scripts/finetune.py --config configs/finetune_tusz.yaml
python scripts/finetune.py --config configs/finetune_somatomotor.yaml
```

## Ablations

```bash
python scripts/run_ablation.py --config configs/ablations/no_sensor_encoder.yaml
python scripts/run_ablation.py --config configs/ablations/no_cross_attention.yaml
python scripts/run_ablation.py --config configs/ablations/signal_only_no_text.yaml
python scripts/run_ablation.py --config configs/ablations/no_orthogonality.yaml
python scripts/run_ablation.py --config configs/ablations/joint_attention.yaml
python scripts/run_ablation.py --config configs/ablations/no_rvq.yaml
python scripts/run_ablation.py --config configs/ablations/no_freq_pcc.yaml
```

## Dataset manifests

The loaders expect JSONL manifests. Each row can point to EDF/FIF/NPY/PT files and optional report text:

```json
{"record_id":"subject001_session001", "signal_path":"/data/tuh/xxx.edf", "report_path":"/data/reports/xxx.txt", "label":1, "dataset":"TUAB", "subject_id":"subject001"}
```

Use the helper scripts as templates:

```bash
python tools/prepare_tuh_eeg.py --root /data/tuh_eeg --out manifests/tuh.jsonl
python tools/prepare_openneuro_somatomotor.py --root /data/openneuro --out manifests/somatomotor.jsonl
```

## Citation

```bibtex
@article{neuroalignemeg2026,
  title={NeuroAlign-EMEG: Sensor-Aware EEG-MEG Foundation Learning with Clinical Report Alignment for Neurodiagnostic Modeling},
  author={Alfraihi, Hessa and Mukhtar, Umar Raza and Mukhtar, Hamza},
  year={2026}
}
```
