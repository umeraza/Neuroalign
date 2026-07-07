from _common import build_loaders, parse_args
from neuroalign_emeg.engine import evaluate
from neuroalign_emeg.models import NeuroAlignEMEG
from neuroalign_emeg.utils import get_device, load_checkpoint, load_yaml, set_seed


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    model = NeuroAlignEMEG(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, strict=False)
    _, val_loader = build_loaders(cfg)
    print(evaluate(model, val_loader, device))


if __name__ == "__main__":
    main()
