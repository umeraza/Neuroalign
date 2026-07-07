from __future__ import annotations

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    torch.save(ckpt.get("model", ckpt), args.out)
    print(f"Exported state_dict to {args.out}")


if __name__ == "__main__":
    main()
