from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuroalign_emeg.data.synthetic import SyntheticEMEGDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--manifest", default="manifests/synthetic.jsonl")
    p.add_argument("--num-samples", type=int, default=32)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    ds = SyntheticEMEGDataset(num_samples=args.num_samples)
    with open(args.manifest, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            sig_path = out_dir / f"sample_{i:05d}.pt"
            rep_path = out_dir / f"sample_{i:05d}.txt"
            torch.save({"signal": item["signal"]}, sig_path)
            rep_path.write_text("\n".join(item["report_segments"]), encoding="utf-8")
            row = {"record_id": item["record_id"], "subject_id": item["subject_id"], "signal_path": str(sig_path), "report_path": str(rep_path), "label": int(item["label"])}
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
