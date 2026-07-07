from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Build a JSONL manifest for OpenNeuro EEG/MEG BIDS exports after local download.")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for f in sorted(list(root.rglob("*.fif")) + list(root.rglob("*.edf"))):
        rid = f.stem
        subj = next((part for part in f.parts if part.startswith("sub-")), rid)
        rows.append({"record_id": rid, "subject_id": subj, "signal_path": str(f), "label": 0, "dataset": "SomatoMotor"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}. Add task labels before training.")


if __name__ == "__main__":
    main()
