# NeuroAlign-EMEG

NeuroAlign-EMEG: Sensor-Aware EEG-MEG Foundation Learning with Clinical Report Alignment for Neurodiagnostic Modeling**.


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

## Dataset manifests

The loaders expect JSONL manifests. Each row can point to EDF/FIF/NPY/PT files and optional report text:

```json
{"record_id":"subject001_session001", "signal_path":"/data/tuh/xxx.edf", "report_path":"/data/reports/xxx.txt", "label":1, "dataset":"TUAB", "subject_id":"subject001"}
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
