# Datasets

The paper evaluates NeuroAlign-EMEG on:

| Role | Dataset | Expected repo adapter |
|---|---|---|
| Signal pretraining | TUEG | `tools/prepare_tuh_eeg.py` + `ManifestEMEGDataset` |
| Report alignment | EEG-Report / ELM-style EEG-report pairs | JSONL rows with `signal_path` + `report_path` |
| Abnormal detection | TUAB | binary labels in manifest |
| Event classification | TUEV | six-class event labels in manifest |
| Seizure/event detection | TUSZ | event labels plus optional onset/offset metadata |
| Cross-modality | SomatoMotor EMEG | local OpenNeuro/BIDS export manifest |

## Manifest schema

```json
{"record_id":"...", "subject_id":"...", "signal_path":"...", "report_path":"...", "label":0, "dataset":"TUAB"}
```

## Leakage rule

All official runs must split by `subject_id`, never by window, because segment-level random splitting leaks patient information.
