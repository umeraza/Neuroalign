from __future__ import annotations

import argparse
import json
from pathlib import Path


def infer_label(path: Path) -> int:
    lower = str(path).lower()
    if "abnormal" in lower or "/ab/" in lower:
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(description="Build a JSONL manifest for TUH EEG-style EDF folders.")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reports-root", default=None)
    args = p.parse_args()
    root = Path(args.root)
    reports_root = Path(args.reports_root) if args.reports_root else None
    rows = []
    for edf in sorted(root.rglob("*.edf")):
        rid = edf.stem
        report_path = None
        if reports_root:
            cand = reports_root / f"{rid}.txt"
            report_path = str(cand) if cand.exists() else None
        rows.append({"record_id": rid, "subject_id": edf.parts[-3] if len(edf.parts) > 3 else rid, "signal_path": str(edf), "report_path": report_path, "label": infer_label(edf), "dataset": "TUH"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
