# Model Card: NeuroAlign-EMEG

## Intended use

Research on EEG/MEG representation learning, sensor-aware tokenization, masked neuro-token modeling, and weak alignment of electrophysiology recordings with clinical text.

## Not intended use

This code is not a medical device. It must not be used for clinical diagnosis, triage, or treatment decisions without external validation, regulatory review, and prospective safety evaluation.

## Inputs

- `signal`: tensor `[channels, time]`
- `sensor_locs`: tensor `[channels, 6]` with XYZ + orientation
- `sensor_types`: tensor `[channels]`
- `report_segments`: optional list of report sentences/sections

## Outputs

- reconstructed waveform
- RVQ token indices
- masked token logits
- signal/report embeddings
- downstream logits

## Known limitations

- Dataset-specific montage correction must be validated manually.
- The default text encoder is a frozen hashing encoder for reproducibility only.
- Public release cannot include restricted clinical reports.
